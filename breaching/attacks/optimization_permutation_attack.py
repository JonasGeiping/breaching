"""Implementation for a subtype of gradient inversion attacks.
Here, all labels/tokens are recovered before optimization starts and the optimization attempts to find the optimal
permutation matrix to match these labels/tokens to the observed data. This makes a lot of sense for language tasks
where tokens are easily leaked from the embedding layer.

Currently only implemented for causal and mlm
"""

import torch
import time

from .optimization_based_attack import OptimizationBasedAttacker

import logging

log = logging.getLogger(__name__)


class OptimizationPermutationAttacker(OptimizationBasedAttacker):
    """Implements an optimization-based attacks that only recovers the order of all tokens.

    TODO: Cut down on the terrible amount of code overlap between this class and OptimizationBasedAttacker"""

    def _recover_token_information(self, user_data, server_payload, rec_models):
        """Recover token information. This is a variation of previous attacks on label recovery, but can abuse
        the embeddings layer in addition to the decoder layer.

        The behavior with respect to multiple queries is work in progress and subject of debate.
        """
        num_data_points = user_data[0]["metadata"]["num_data_points"]
        num_classes = user_data[0]["gradients"][-1].shape[0]
        num_queries = len(user_data)

        if self.cfg.label_strategy == "decoder-bias":
            # works super well for normal stuff like transformer3 without tying
            num_missing_tokens = num_data_points * self.data_shape[0]
            # This is slightly modified analytic label recovery in the style of Wainakh
            # have to assert that this is the decoder bias:
            bias_per_query = [shared_data["gradients"][-1] for shared_data in user_data]
            assert len(bias_per_query[0]) == server_payload[0]["metadata"]["vocab_size"]

            token_list = []
            # Stage 1
            average_bias = torch.stack(bias_per_query).mean(dim=0)
            valid_classes = (average_bias < 0).nonzero()
            token_list += [*valid_classes.squeeze(dim=-1)]
            # Supplement with missing tokens from input:
            tokens_in_input = self.embeddings[0]["grads"].norm(dim=-1).nonzero().squeeze(dim=-1)
            for token in tokens_in_input:
                if token not in token_list:
                    token_list.append(token)

            m_impact = average_bias_correct_label = average_bias[valid_classes].sum() / num_missing_tokens

            average_bias[valid_classes] = average_bias[valid_classes] - m_impact
            # Stage 2
            while len(label_list) < num_missing_tokens:
                selected_idx = average_bias.argmin()
                tokens_list.append(selected_idx)
                average_bias[selected_idx] -= m_impact
            tokens = torch.stack(token_list).view(num_data_points, self.data_shape[0])
        elif self.cfg.label_strategy == "embedding-norm":
            # This works decently well for GPT which has no decoder bias
            wte_per_query = [shared_data["gradients"][0] for shared_data in user_data]
            assert wte_per_query[0].shape[0] == server_payload[0]["metadata"]["vocab_size"]

            token_list = []
            # Stage 1
            average_wte_norm = torch.stack(wte_per_query).mean(dim=0).norm(dim=1)
            std, mean = torch.std_mean(average_wte_norm.log())
            cutoff = mean + 3 * std
            valid_classes = (average_wte_norm.log() > cutoff).nonzero()
            token_list += [*valid_classes.squeeze(dim=-1)]

            # top2-log rule is decent:
            # top2 = average_wte_norm.log().topk(k=2).values  # log here is not an accident!
            # m_impact = top2[0] - top2[1]
            # but the sum is simpler:
            m_impact = average_wte_norm[valid_classes].sum() / num_missing_tokens

            average_wte_norm[valid_classes] = average_wte_norm[valid_classes] - m_impact
            # Stage 2
            while len(token_list) < num_missing_tokens:
                selected_idx = valid_classes[average_wte_norm[valid_classes].argmax()].squeeze()
                token_list.append(selected_idx)
                # print(selected_idx, average_wte_norm_log[selected_idx])
                average_wte_norm[selected_idx] -= m_impact
            tokens = torch.stack(token_list).view(num_data_points, data_shape[0])

        elif self.cfg.label_strategy == "mixed":
            # Can improve performance for tied embeddings over just decoder-bias
            # as unique token extraction is slightly more exact from the embedding layer
            wte_per_query = [shared_data["gradients"][0]]
            assert wte_per_query[0].shape[0] == server_payload[0]["metadata"]["vocab_size"]
            average_wte_norm = torch.stack(wte_per_query).mean(dim=0).norm(dim=1)

            bias_per_query = [shared_data["gradients"][-1]]
            assert len(bias_per_query[0]) == server_payload[0]["metadata"]["vocab_size"]
            average_bias = torch.stack(bias_per_query).mean(dim=0)

            token_list = []
            # Stage 1
            std, mean = torch.std_mean(average_wte_norm.log())
            cutoff = mean + 2.5 * std
            valid_classes = (average_wte_norm.log() > cutoff).nonzero()
            token_list += [*valid_classes.squeeze(dim=-1)]

            m_impact = average_bias[valid_classes].sum() / num_missing_tokens
            average_bias[valid_classes] = average_bias[valid_classes] - m_impact
            # Stage 2
            while len(token_list) < num_missing_tokens:
                selected_idx = valid_classes[average_bias[valid_classes].argmin()].squeeze()
                token_list.append(selected_idx)
                average_bias[selected_idx] -= m_impact
            tokens = torch.stack(token_list).view(num_data_points, data_shape[0])

        else:
            raise ValueError(f"Invalid label strategy {self.cfg.label_strategy} for attack_type permutation attack.")

        # Always sort, order does not matter here:
        labels = labels.sort()[0]
        log.info(f"Recovered tokens {tokens.tolist()} through strategy {self.cfg.label_strategy}.")
        return labels

    def _postprocess_text_data(self, reconstructed_user_data):
        """Post-processsing text data to recover tokens is not necessary here.
           Instead we recover the argmax assignment of the permutation matrix."""
        from scipy.optimize import linear_sum_assignment  # Again a lazy import

        _, rec_assignment = linear_sum_assignment(reconstructed_user_data["data"].cpu().numpy(), maximize=True)
        reconstructed_user_data = dict(data=rec_assignment, labels=rec_assignment)
        return reconstructed_user_data

    def _run_trial(self, rec_model, shared_data, tokens, stats, trial, initial_data=None, dryrun=False):
        """Run a single reconstruction trial."""

        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, tokens)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])

        # Initialize permutation_matrix reconstruction data
        possible_positions = shared_data[0]["metadata"]["num_data_points"] * self.data_shape[0]
        self.embedding_shape = [shared_data[0]["metadata"]["num_data_points"], *self.data_shape]
        # Disregard actual data shape, initialize a permutation matrix:
        permutation_matrix = self._initialize_data([possible_positions, possible_positions])
        if initial_data is not None:
            permutation_matrix.data = initial_data.data.clone().to(**self.setup)

        best_permutation_matrix = permutation_matrix.detach().clone()
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

        # Initialize optimizers
        optimizer, scheduler = self._init_optimizer([permutation_matrix])
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective(
                    permutation_matrix, tokens, rec_model, optimizer, shared_data, iteration
                )
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                scheduler.step()

                with torch.no_grad():
                    # Project onto permutation matrix:
                    permutation_matrix.data.clamp_(0, 1)
                    permutation_matrix.data = self._sinkhorn_knopp(permutation_matrix.data)
                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_permutation_matrix = permutation_matrix.detach().clone()

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    log.info(
                        f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                    )
                    current_wallclock = timestamp

                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass

        return best_permutation_matrix.detach()

    def _sinkhorn_knopp(self, permutation_matrix, sub_iterations=50, reg=0.01):
        """Project onto doube-stochastic matrices.
        Could mostly stolen from https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn
        Could also do https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn_log?
        """
        ndim = permutation_matrix.shape[0]
        u, v, a, b = [torch.ones(ndim, **self.setup) / ndim] * 4

        K = torch.candidate_data(M / (-reg))
        Kp = (1 / a).reshape(-1, 1) * K

        for iteration in range(sub_iterations):
            uprev = u
            vprev = v
            KtransposeU = torch.dot(K.T, u)
            v = b / KtransposeU
            u = 1.0 / torch.dot(Kp, v)

            if (
                torch.any(KtransposeU == 0)
                or torch.any(torch.isnan(u))
                or torch.any(torch.isnan(v))
                or torch.any(torch.isinf(u))
                or torch.any(torch.isinf(v))
            ):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                warnings.warn("Warning: numerical errors at iteration %d" % ii)
                u = uprev
                v = vprev
                break
        return u.reshape((-1, 1)) * K * v.reshape((1, -1))

    def _compute_objective(self, permutation_matrix, tokens, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()

            total_objective = 0
            total_task_loss = 0

            for model, data, embedding in zip(rec_model, shared_data, self.embeddings):
                current_embedding = permutation_matrix.matmul(embedding["weight"][tokens]).view(self.embedding_shape)
                soft_labels = torch.zeros([*self.embedding_shape[:-1], embedding["weight"].shape[0]], **setup)
                soft_labels[tokens] = permutation_matrix.softmax(dim=-1)
                objective, task_loss = self.objective(model, data["gradients"], current_embedding, soft_labels)
                total_objective += objective
                total_task_loss += task_loss
                for regularizer in self.regularizers:
                    total_objective += regularizer(current_embedding)

            if total_objective.requires_grad:
                total_objective.backward(inputs=permutation_matrix, create_graph=False)
            with torch.no_grad():
                if self.cfg.optim.langevin_noise > 0:
                    step_size = optimizer.param_groups[0]["lr"]
                    noise_map = torch.randn_like(permutation_matrix.grad)
                    permutation_matrix.grad += self.cfg.optim.langevin_noise * step_size * noise_map
                if self.cfg.optim.grad_clip is not None:
                    grad_norm = permutation_matrix.grad.norm()
                    if grad_norm > self.cfg.optim.grad_clip:
                        permutation_matrix.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                if self.cfg.optim.signed is not None:
                    if self.cfg.optim.signed == "soft":
                        scaling_factor = (
                            1 - iteration / self.cfg.optim.max_iterations
                        )  # just a simple linear rule for now
                        permutation_matrix.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                    elif self.cfg.optim.signed == "hard":
                        permutation_matrix.grad.sign_()
                    else:
                        pass

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure
