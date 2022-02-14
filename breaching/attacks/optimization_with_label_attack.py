"""Implementation for basic gradient inversion attacks. This variation does not compute labels
by a fixed formula before optimization, but optimizes jointly for candidate data and labels
(as in the original DLG paper). This is usually a bad idea as the optimization landscape behaves
even worse, but is necessary in some scenarios (like text, where label order matters).
And it does also work in simpler settings as in the original paper.
In some cases turning this on can stabilize an L-BFGS optimizer.
"""

import torch
import time

from .optimization_based_attack import OptimizationBasedAttacker
from .auxiliaries.regularizers import TotalVariation
from .auxiliaries.objectives import Euclidean, CosineSimilarity

import logging

log = logging.getLogger(__name__)


class CandidateDict(dict):
    """Container for a candidate solution. Behaves like a torch.Tensor?"""

    def __init__(self, tensor_dict, *args, **kwargs):
        self.tensor_dict = tensor_dict

    def __getitem__(self, key):
        return self.tensor_dict[key]

    def __getattr__(self, name):
        return_vals = CandidateDict(dict())
        for key, tensor in self.tensor_dict.items():
            return_vals[key] = getattr(tensor, name)
        return return_vals


class OptimizationJointAttacker(OptimizationBasedAttacker):
    """Implements a wide spectrum of optimization-based attacks optimizing jointly for candidate data and labels.

    TODO: Cut down on the terrible amount of code overlap between this class and OptimizationBasedAttacker"""

    def _recover_label_information(self, user_data, server_payload, rec_models, embedding_grads=None):
        num_data_points = user_data[0]["metadata"]["num_data_points"]
        metadata = server_payload[0]["metadata"]
        if metadata["task"] == "classification":
            label_candidate = self._initialize_data([num_data_points, metadata.classes])
        else:  # segmentation type in_shape->out_shape tasks
            label_candidate = self._initialize_data([num_data_points, self.data_shape[0], metadata.vocab_size])
        return label_candidate

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        if shared_data[0]["metadata"]["labels"] is not None:
            raise ValueError(
                "Joint optimization only makes sense if no labels are provided. "
                "Switch to attack.attack_type=optimization instead"
            )

        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions, candidate_labels = [], []
        try:
            for trial in range(self.cfg.restarts.num_trials):
                data, label = self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun)
                candidate_solutions += [data]
                candidate_labels += [labels.argmax(dim=-1)]
                scores[trial] = self._score_trial(
                    candidate_solutions[trial], candidate_labels[trial], rec_models, shared_data
                )
        except KeyboardInterrupt:
            print("Trial procedure manually interruped.")
            pass
        optimal_solution, optimal_labels = self._select_optimal_reconstruction(
            candidate_solutions, candidate_labels, scores, stats
        )
        reconstructed_data = dict(data=optimal_solution, labels=optimal_labels)
        if server_payload[0]["metadata"].modality == "text":
            reconstructed_data = self._postprocess_text_data(reconstructed_data)
            reconstructed_data["raw_embeddings"] = optimal_solution
        if "ClassAttack" in server_secrets:
            # Only a subset of images was actually reconstructed:
            true_num_data = server_secrets["ClassAttack"]["true_num_data"]
            reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
            reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = optimal_solution
            reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
        return reconstructed_data, stats

    def _run_trial(self, rec_model, shared_data, label_template, stats, trial, initial_data=None, dryrun=False):
        """Run a single reconstruction trial."""

        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])

        # Initialize candidate reconstruction data
        candidate_data = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
        candidate_labels = self._initialize_data(label_template.shape)
        if initial_data is not None:
            candidate_data.data = initial_data.data.clone().to(**self.setup)

        best_candidate = candidate_data.detach().clone()
        best_labels = candidate_labels.detach().clone()
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

        # Initialize optimizers
        optimizer, scheduler = self._init_optimizer([candidate_data, candidate_labels])
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective(
                    candidate_data, candidate_labels, rec_model, optimizer, shared_data, iteration
                )
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.cfg.optim.boxed:
                        candidate_data.data = torch.max(
                            torch.min(candidate_data, (1 - self.dm) / self.ds), -self.dm / self.ds
                        )
                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_candidate = candidate_data.detach().clone()
                        best_labels = candidate_labels.detach().clone()

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    p = candidate_labels.softmax(dim=-1)
                    label_entropy = torch.where(p > 0, -p * torch.log(p), torch.zeros_like(p),).sum(
                        dim=-1
                    ).mean() / torch.log(torch.as_tensor(p.shape[-1], dtype=torch.float))
                    log.info(
                        f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s | "
                        f" Label Entropy: {label_entropy:2.4f}."
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

        return best_candidate.detach(), best_labels.detach()

    def _compute_objective(self, candidate, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()

            if self.cfg.differentiable_augmentations:
                candidate_augmented = self.augmentations(candidate)
            else:
                candidate_augmented = candidate
                candidate_augmented.data = self.augmentations(candidate.data)
            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):
                objective, task_loss = self.objective(
                    model, data["gradients"], candidate_augmented, labels.softmax(dim=-1)
                )
                total_objective += objective
                total_task_loss += task_loss
            for regularizer in self.regularizers:
                total_objective += regularizer(candidate_augmented)

            if total_objective.requires_grad:
                total_objective.backward(inputs=[candidate, labels], create_graph=False)
            with torch.no_grad():
                if self.cfg.optim.langevin_noise > 0:
                    step_size = optimizer.param_groups[0]["lr"]
                    noise_map = torch.randn_like(candidate.grad)
                    candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
                    labels.grad += self.cfg.optim.langevin_noise * step_size * torch.randn_like(labels.grad)
                if self.cfg.optim.grad_clip is not None:
                    for element in [candidate, labels]:
                        grad_norm = element.grad.norm()
                        if grad_norm > self.cfg.optim.grad_clip:
                            element.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                if self.cfg.optim.signed is not None:
                    if self.cfg.optim.signed == "soft":
                        scaling_factor = (
                            1 - iteration / self.cfg.optim.max_iterations
                        )  # just a simple linear rule for now
                        candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                        labels.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                    elif self.cfg.optim.signed == "hard":
                        candidate.grad.sign_()
                        labels.grad.sign_()
                    else:
                        pass

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure

    def _score_trial(self, candidate, labels, rec_model, shared_data):
        """Score candidate solutions based on some criterion."""

        if self.cfg.restarts.scoring in ["euclidean", "cosine-similarity"]:
            objective = Euclidean() if self.cfg.restarts.scoring == "euclidean" else CosineSimilarity()
            objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
            score = 0
            for model, data in zip(rec_model, shared_data):
                score += objective(model, data["gradients"], candidate, labels)[0]
        elif self.cfg.restarts.scoring in ["TV", "total-variation"]:
            score = TotalVariation(scale=1.0)(candidate)
        else:
            raise ValueError(f"Scoring mechanism {self.cfg.scoring} not implemented.")
        return score if score.isfinite() else float("inf")

    def _select_optimal_reconstruction(self, candidate_solutions, candidate_labels, scores, stats):
        """Choose one of the candidate solutions based on their scores (for now).

        More complicated combinations are possible in the future."""
        optimal_val, optimal_index = torch.min(scores, dim=0)
        optimal_solution = candidate_solutions[optimal_index]
        optimal_labels = candidate_labels[optimal_index]
        stats["opt_value"] = optimal_val.item()
        if optimal_val.isfinite():
            log.info(f"Optimal candidate solution with rec. loss {optimal_val.item():2.4f} selected.")
            return optimal_solution, optimal_labels
        else:
            log.info("No valid reconstruction could be found.")
            return torch.zeros_like(optimal_solution), torch.zeros_like(optimal_labels)
