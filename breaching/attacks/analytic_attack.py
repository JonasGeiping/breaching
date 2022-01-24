"""Simple analytic attack that works for (dumb) fully connected models."""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment  # Better than greedy search


from collections import defaultdict

from .base_attack import _BaseAttacker
from ..cases.models.transformer_dictionary import lookup_module_names

import logging

log = logging.getLogger(__name__)


class AnalyticAttacker(_BaseAttacker):
    """Implements a sanity-check analytic inversion

    Only works for a torch.nn.Sequential model with input-sized FC layers."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)

    def __repr__(self):
        return f"""Attacker (of type {self.__class__.__name__})."""

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        # Main reconstruction: loop starts here:
        inputs_from_queries = []
        for model, user_data in zip(rec_models, shared_data):
            idx = len(user_data["gradients"]) - 1
            for layer in list(model)[::-1]:  # Only for torch.nn.Sequential
                if isinstance(layer, torch.nn.Linear):
                    bias_grad = user_data["gradients"][idx]
                    weight_grad = user_data["gradients"][idx - 1]
                    layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, labels)
                    idx -= 2
                elif isinstance(layer, torch.nn.Flatten):
                    inputs = layer_inputs.reshape(user_data["metadata"]["num_data_points"], *self.data_shape)
                else:
                    raise ValueError(f"Layer {layer} not supported for this sanity-check attack.")
            inputs_from_queries += [inputs]

        final_reconstruction = torch.stack(inputs_from_queries).mean(dim=0)
        reconstructed_data = dict(data=inputs, labels=labels)

        return reconstructed_data, stats

    def invert_fc_layer(self, weight_grad, bias_grad, image_positions):
        """The basic trick to invert a FC layer."""
        # By the way the labels are exactly at (bias_grad < 0).nonzero() if they are unique
        valid_classes = bias_grad != 0
        intermediates = weight_grad[valid_classes, :] / bias_grad[valid_classes, None]
        if len(image_positions) == 0:
            reconstruction_data = intermediates
        elif len(image_positions) == 1:
            reconstruction_data = intermediates.mean(dim=0)
        else:
            reconstruction_data = intermediates[image_positions]
        return reconstruction_data


class ImprintAttacker(AnalyticAttacker):
    """Abuse imprint secret for near-perfect attack success."""

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """This is somewhat hard-coded for images, but that is not a necessity."""
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        len_data = shared_data[0]["metadata"]["num_data_points"]  # Not strictly needed for the attack, used to pad/trim

        if "ImprintBlock" in server_secrets.keys():
            weight_idx = server_secrets["ImprintBlock"]["weight_idx"]
            bias_idx = server_secrets["ImprintBlock"]["bias_idx"]
            data_shape = server_secrets["ImprintBlock"]["shape"]
        else:
            raise ValueError(f"No imprint hidden in model {rec_models[0]} according to server.")

        bias_grad = shared_data[0]["gradients"][bias_idx].clone()
        weight_grad = shared_data[0]["gradients"][weight_idx].clone()

        if self.cfg.sort_by_bias:
            # This variant can recover from shuffled rows under the assumption that biases would be ordered
            _, order = server_payload[0]["parameters"][1].sort(descending=True)
            bias_grad = bias_grad[order]
            weight_grad = weight_grad[order]

        if server_secrets["ImprintBlock"]["structure"] == "cumulative":
            for i in reversed(list(range(1, weight_grad.shape[0]))):
                weight_grad[i] -= weight_grad[i - 1]
                bias_grad[i] -= bias_grad[i - 1]

        image_positions = bias_grad.nonzero()
        layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, [])

        if "decoder" in server_secrets["ImprintBlock"].keys():
            inputs = server_secrets["ImprintBlock"]["decoder"](layer_inputs)
        else:
            inputs = layer_inputs.reshape(layer_inputs.shape[0], *data_shape)[:, :3, :, :]
        if weight_idx > 0:  # An imprint block later in the network:
            inputs = torch.nn.functional.interpolate(
                inputs, size=self.data_shape[1:], mode="bicubic", align_corners=False
            )
        inputs = torch.max(torch.min(inputs, (1 - self.dm) / self.ds), -self.dm / self.ds)

        if len_data >= inputs.shape[0]:
            # Fill up with zero if not enough data can be found:
            missing_entries = torch.zeros(len_data - inputs.shape[0], *self.data_shape, **self.setup)
            inputs = torch.cat([inputs, missing_entries], dim=0)
        else:
            print(f"Initially produced {inputs.shape[0]} hits.")
            # Cut additional hits:
            # this rule is optimal for clean data with few bins:
            # best_guesses = torch.topk(bias_grad[bias_grad != 0].abs(),len_data, largest=False)
            # this rule is best when faced with differential privacy:
            best_guesses = torch.topk(weight_grad.mean(dim=1)[bias_grad != 0].abs(), len_data, largest=True)
            print(f"Reduced to {len_data} hits.")
            # print(best_guesses.indices.sort().values)
            inputs = inputs[best_guesses.indices]

        reconstructed_data = dict(data=inputs, labels=labels)
        return reconstructed_data, stats


class DecepticonAttacker(AnalyticAttacker):
    """An analytic attack against transformer models in language."""

    @torch.inference_mode()
    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """Reconstruct both positions and token ids from the input sentence. Disambiguate sentences based on v_length."""
        # Initialize stats module for later usage:
        rec_models, tokens, stats = self.prepare_attack(server_payload, shared_data)
        len_data = shared_data[0]["metadata"]["num_data_points"]  # Could be guessed as well
        lookup = lookup_module_names(rec_models[0].name, rec_models[0])

        if "ImprintBlock" in server_secrets.keys():
            weight_idx = server_secrets["ImprintBlock"]["weight_idx"]
            bias_idx = server_secrets["ImprintBlock"]["bias_idx"]
            data_shape = server_secrets["ImprintBlock"]["data_shape"]
            v_length = server_secrets["ImprintBlock"]["v_length"]
            ff_transposed = server_secrets["ImprintBlock"]["ff_transposed"]
        else:
            raise ValueError(f"No imprint hidden in model {rec_models[0]} according to server.")
        [model.eval() for model in rec_models]

        leaked_tokens = self.recover_token_information(shared_data, server_payload, rec_models[0].name).view(-1)
        leaked_embeddings = lookup["norm_layer1"](lookup["embedding"](leaked_tokens))
        leaked_embeddings = leaked_embeddings.cpu().view(-1, lookup["embedding"].weight.shape[1])

        bias_grad = torch.cat([shared_data[0]["gradients"][b_idx].clone() for b_idx in bias_idx])
        if ff_transposed:
            weight_grad = torch.cat([shared_data[0]["gradients"][w_idx].clone() for w_idx in weight_idx], dim=1)
            weight_grad = weight_grad.T.contiguous()  # Restride only due to paranoia
        else:
            weight_grad = torch.cat([shared_data[0]["gradients"][w_idx].clone() for w_idx in weight_idx])

        if self.cfg.sort_by_bias:
            # This variant can recover from shuffled rows under the assumption that biases would be ordered
            _, order = server_payload[0]["parameters"][1].sort(descending=True)
            bias_grad = bias_grad[order]
            weight_grad = weight_grad[order]

        if server_secrets["ImprintBlock"]["structure"] == "cumulative":
            for i in range(0, weight_grad.shape[0] - 1):
                weight_grad[i] -= weight_grad[i + 1]
                bias_grad[i] -= bias_grad[i + 1]

        # Here are our reconstructed positionally encoded embeddings:
        valid_classes = bias_grad != 0
        breached_embeddings = weight_grad[valid_classes, :] / bias_grad[valid_classes, None]
        log.info(f"Recovered {len(breached_embeddings)} embeddings with positional data from imprinted layer.")
        if len(breached_embeddings) > len_data * data_shape[0]:
            best_guesses = torch.topk(
                weight_grad.mean(dim=1)[bias_grad != 0].abs(), len_data * data_shape[0], largest=True
            )
            best_guesses = torch.topk(bias_grad[bias_grad != 0].abs(), len_data * data_shape[0], largest=False)
            log.info(f"Reduced to {len_data * data_shape[0]} hits.")
            # print(best_guesses.indices.sort().values)
            breached_embeddings = breached_embeddings[best_guesses.indices]
        breached_embeddings = breached_embeddings.cpu()  # Assignments run on CPU anyway
        if (~torch.isfinite(breached_embeddings)).sum():
            raise ValueError("Invalid breached embeddings recovered.")

        # Get an estimation of the positional embeddings:
        dummy_inputs = torch.zeros([len_data, *data_shape], dtype=torch.long, device=self.setup["device"])
        pure_positions = lookup["pos_encoder"](torch.zeros_like(lookup["embedding"](dummy_inputs)))
        pure_normed_positions = lookup["norm_layer1"](pure_positions)
        positional_embeddings = pure_normed_positions.cpu().view(-1, lookup["embedding"].weight.shape[1])

        # Step 0: Separate breached embeddings into separate sentences:
        sentence_id_components = breached_embeddings[:, :v_length]

        if len_data > 1:
            sentence_labels = self._match_breaches_to_sentences(
                sentence_id_components, [len_data, data_shape[0]], algorithm=self.cfg.sentence_algorithm
            )
            _, counts = sentence_labels.unique(return_counts=True)
            log.info(f"Assigned {counts.tolist()} breached embeddings to each sentence.")
        else:
            sentence_labels = torch.zeros(len(breached_embeddings), dtype=torch.long)

        # Match breached embeddings to positions for each sentence:
        breached_embeddings = breached_embeddings[:, v_length:-1]
        positional_embeddings = positional_embeddings[:, v_length:-1]
        leaked_embeddings = leaked_embeddings[:, v_length:-1]

        ordered_breached_embeddings = torch.zeros_like(positional_embeddings)
        for sentence in range(len_data):
            order_breach_to_positions, costs = self._match_embeddings(
                positional_embeddings[: data_shape[0]], breached_embeddings[sentence_labels == sentence]
            )
            ordered_breached_embeddings[sentence * data_shape[0] + order_breach_to_positions] = breached_embeddings[
                sentence_labels == sentence
            ]
        # Then fill up the missing locations:
        if len(breached_embeddings) < len(positional_embeddings):
            free_positions = (ordered_breached_embeddings.norm(dim=-1) == 0).nonzero().squeeze(dim=1)
            assert len(breached_embeddings) > len(free_positions)
            miss_to_pos, costs = self._match_embeddings(breached_embeddings, positional_embeddings[free_positions])
            ordered_breached_embeddings[free_positions] = breached_embeddings[miss_to_pos]

        # These are already in the right position, but which token do they belong to?
        # b_std, b_mean = torch.std_mean(ordered_breached_embeddings, dim=-1, keepdim=True)
        # normalized_ordered_breached = (ordered_breached_embeddings - b_mean) / (b_std + 1e-10)
        # p_std, p_mean = torch.std_mean(positional_embeddings, dim=-1, keepdim=True)
        # normalized_positions = (positional_embeddings - p_mean) / (p_std + 1e-10)
        # breached_without_positions = normalized_ordered_breached - normalized_positions
        # A = torch.stack([ordered_breached_embeddings.view(-1, 89), positional_embeddings.view(-1, 89)], dim=-1)
        # U, S, V = torch.pca_lowrank(A, q=None, center=True, niter=2,)
        # breached_without_positions = torch.matmul(A, V[:, 1:2].permute(0, 2, 1)).view_as(ordered_breached_embeddings)
        breached_without_positions = ordered_breached_embeddings - positional_embeddings
        order_leaked_to_breached, costs = self._match_embeddings(leaked_embeddings, breached_without_positions)
        recovered_tokens = leaked_tokens[order_leaked_to_breached]
        if self.cfg.embedding_token_weight > 0:
            # Optionally: Match breached_without_positions to any embedding entries
            # If the costs from the matching above are low, then this can recover lost tokens that were missed by
            # .recover_token_information()
            vocab_size = server_payload[0]["metadata"]["vocab_size"]
            all_token_ids = torch.arange(0, vocab_size, device=self.setup["device"])
            all_token_embeddings = lookup["norm_layer1"](lookup["embedding"](all_token_ids))
            all_token_embeddings = all_token_embeddings[:, v_length:-1]
            avg_costs = 0
            avg_new_corr = 0
            num_replaced_tokens = 0
            for idx, entry in enumerate(breached_without_positions.detach().cpu().numpy()):
                max_corr = self.vcorrcoef(all_token_embeddings.detach()[1:].cpu().numpy(), entry)
                val, loc = torch.as_tensor(max_corr).max(dim=0)
                if val * self.cfg.embedding_token_weight > costs[idx]:
                    num_replaced_tokens += 1
                    avg_costs += costs[idx]
                    avg_new_corr += val
                    recovered_tokens[idx] = loc + 1
            log.info(
                f"Replaced {num_replaced_tokens} tokens with avg. corr {avg_costs / num_replaced_tokens} "
                f"with new tokens with avg corr {avg_new_corr / num_replaced_tokens}"
            )
        # Finally re-order into sentences:
        final_tokens = recovered_tokens.view([len_data, *data_shape])
        reconstructed_data = dict(data=final_tokens, labels=final_tokens)
        return reconstructed_data, stats

    def _match_breaches_to_sentences(self, sentence_id_components, shape, algorithm="threshold"):
        if algorithm == "k-means":
            from k_means_constrained import KMeansConstrained

            clustering = KMeansConstrained(
                n_clusters=shape[0], size_min=0, size_max=shape[1], init="k-means++", n_init=40, max_iter=900, tol=1e-6,
            )
            std, mean = torch.std_mean(sentence_id_components, dim=-1, keepdim=True)
            normalized_components = (sentence_id_components - mean) / (std + 1e-8)
            labels = clustering.fit_predict(sentence_id_components.numpy())
            sentence_labels = torch.as_tensor(labels)

        elif algorithm == "dynamic-threshold":
            corrs = torch.as_tensor(np.corrcoef(sentence_id_components.double().detach().numpy()))
            corrs[torch.isnan(corrs)] = 0  # Should only ever trigger in edge cases where sentence_component=0
            upper_range = [1 - 1.5 ** float(n) for n in torch.arange(-96, -16)][::-1]
            lower_range = 1.001 - np.geomspace(1, 0.001, 2000)[:-1]
            trial_tresholds = [*lower_range, *upper_range]
            num_entries = []
            for idx, threshold in enumerate(trial_tresholds[::-1]):
                num_entries = (corrs > threshold).sum(dim=-1).max()
                # print(idx, threshold, num_entries, shape[1])
                if num_entries > shape[1]:
                    final_threshold = trial_tresholds[::-1][idx - 1]
                    break
            else:
                log.info(f"Cannot separate {shape[0]} seeds by thresholding!")
                final_threshold = trial_tresholds[0]

            already_assigned = set()
            initial_labels = -torch.ones(corrs.shape[0], dtype=torch.long)
            total_groups = 0
            for idx in (corrs > final_threshold).sum(dim=-1).sort(descending=True).indices:
                if idx.item() not in already_assigned:
                    matches = (corrs[idx] > final_threshold).nonzero().squeeze(0)
                    if len(matches) > 0:
                        total_groups += 1
                        filtered_matches = torch.as_tensor([m for m in matches if m not in already_assigned])
                        initial_labels[filtered_matches] = idx
                        already_assigned |= set(filtered_matches.tolist())
                    if total_groups == shape[0]:
                        break
            if total_groups < shape[0]:
                log.info(f"Could assemble only {total_groups} seeds searching on threshold {final_threshold}.")
                log.info(f"Filling with {shape[0] - total_groups} random seeds...These sentences will be scrambled.")
            # Find seeds
            seeds = torch.randn(shape[0], sentence_id_components.shape[-1])  # seeds for every sentence
            label_ids = initial_labels[initial_labels != -1].unique()  # Skip over -1 here
            for idx, group_label in enumerate(label_ids):
                seeds[idx] = sentence_id_components[initial_labels == group_label].mean(dim=0)
            replicated_seeds = torch.repeat_interleave(seeds, shape[1], dim=0)  # Replicate seeds to seq_length
            # Recompute correlations based on these mean seeds

            order_breach_to_seed, _ = self._match_embeddings(
                replicated_seeds, sentence_id_components, fallbacks=initial_labels
            )
            sentence_labels = (order_breach_to_seed / shape[1]).to(dtype=torch.long)

        elif algorithm == "threshold":
            corrs = torch.as_tensor(np.corrcoef(sentence_id_components.contiguous().detach().numpy()))
            sentence_labels = -torch.ones(corrs.shape[0], dtype=torch.long)
            already_assigned = set()
            for idx in range(corrs.shape[0]):
                if idx not in already_assigned:
                    matches = (corrs[idx] >= 0.99).nonzero().squeeze(0)
                    if len(matches) > 0:
                        filtered_matches = torch.as_tensor([m for m in matches if m not in already_assigned])
                        if len(filtered_matches) > shape[1]:
                            filtered_matches = corrs[idx][filtered_matches].topk(k=shape[1]).indices
                        sentence_labels[filtered_matches] = idx
                        already_assigned |= set(filtered_matches.tolist())
            assert sentence_labels.min() == 0
        elif algorithm == "fcluster":
            import scipy.cluster.hierarchy as spc
            from scipy.spatial.distance import squareform

            corrs = np.corrcoef(sentence_id_components.contiguous().detach().numpy())
            dissimilarity = 1 - np.abs((corrs + corrs.T) / 2)
            np.fill_diagonal(dissimilarity, 0)
            hierarchy = spc.linkage(squareform(dissimilarity), method="ward")
            idx = spc.fcluster(hierarchy, shape[0], criterion="maxclust")
            sentence_labels = torch.as_tensor(idx, dtype=torch.long)
            assert sentence_labels.unique(return_counts=True)[1].max() <= shape[1], "Invalid Assignment in fcluster"
        else:
            raise ValueError(f"Invalid sentence algorithm {algorithm} given.")
        return sentence_labels

    @torch.inference_mode()
    def reconstruct_token_first(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """Reconstruct both positions and token ids from the input sentence. Disambiguate sentences based on v_length."""
        # Initialize stats module for later usage:
        rec_models, tokens, stats = self.prepare_attack(server_payload, shared_data)
        len_data = shared_data[0]["metadata"]["num_data_points"]  # Could be guessed as well
        lookup = lookup_module_names(rec_models[0].name, rec_models[0])

        if "ImprintBlock" in server_secrets.keys():
            weight_idx = server_secrets["ImprintBlock"]["weight_idx"]
            bias_idx = server_secrets["ImprintBlock"]["bias_idx"]
            data_shape = server_secrets["ImprintBlock"]["data_shape"]
            v_length = server_secrets["ImprintBlock"]["v_length"]
            ff_transposed = server_secrets["ImprintBlock"]["ff_transposed"]
        else:
            raise ValueError(f"No imprint hidden in model {rec_models[0]} according to server.")
        [model.eval() for model in rec_models]

        leaked_tokens = self.recover_token_information(shared_data, server_payload, rec_models[0].name).view(-1)
        leaked_embeddings = lookup["norm_layer1"](lookup["embedding"](leaked_tokens))
        leaked_embeddings = leaked_embeddings.cpu().view(-1, lookup["embedding"].weight.shape[1])

        bias_grad = torch.cat([shared_data[0]["gradients"][b_idx].clone() for b_idx in bias_idx])
        if ff_transposed:
            weight_grad = torch.cat([shared_data[0]["gradients"][w_idx].clone() for w_idx in weight_idx], dim=1)
            weight_grad = weight_grad.T.contiguous()  # Restride only due to paranoia
        else:
            weight_grad = torch.cat([shared_data[0]["gradients"][w_idx].clone() for w_idx in weight_idx])

        if self.cfg.sort_by_bias:
            # This variant can recover from shuffled rows under the assumption that biases would be ordered
            _, order = server_payload[0]["parameters"][1].sort(descending=True)
            bias_grad = bias_grad[order]
            weight_grad = weight_grad[order]

        if server_secrets["ImprintBlock"]["structure"] == "cumulative":
            for i in range(0, weight_grad.shape[0] - 1):
                weight_grad[i] -= weight_grad[i + 1]
                bias_grad[i] -= bias_grad[i + 1]

        # Here are our reconstructed positionally encoded embeddings:
        valid_classes = bias_grad != 0
        breached_embeddings = weight_grad[valid_classes, :] / bias_grad[valid_classes, None]
        log.info(f"Recovered {len(breached_embeddings)} embeddings with positional data from imprinted layer.")
        if len(breached_embeddings) > len_data * data_shape[0]:
            best_guesses = torch.topk(
                weight_grad.mean(dim=1)[bias_grad != 0].abs(), len_data * data_shape[0], largest=True
            )
            best_guesses = torch.topk(bias_grad[bias_grad != 0].abs(), len_data * data_shape[0], largest=False)
            log.info(f"Reduced to {len_data * data_shape[0]} hits.")
            # print(best_guesses.indices.sort().values)
            breached_embeddings = breached_embeddings[best_guesses.indices]
        breached_embeddings = breached_embeddings.cpu()  # Assignments run on CPU anyway
        if (~torch.isfinite(breached_embeddings)).sum():
            raise ValueError("Invalid breached embeddings recovered.")

        # Get an estimation of the positional embeddings:
        dummy_inputs = torch.zeros([len_data, *data_shape], dtype=torch.long, device=self.setup["device"])
        pure_positions = lookup["pos_encoder"](torch.zeros_like(lookup["embedding"](dummy_inputs)))
        pure_normed_positions = lookup["norm_layer1"](pure_positions)
        positional_embeddings = pure_normed_positions.cpu().view(-1, lookup["embedding"].weight.shape[1])

        # Step 0: Separate breached embeddings into separate sentences:
        sentence_id_components = breached_embeddings[:, :v_length]

        if len_data > 1:
            sentence_labels = self._match_breaches_to_sentences(
                sentence_id_components, [len_data, data_shape[0]], algorithm=self.cfg.sentence_algorithm
            )
            _, counts = sentence_labels.unique(return_counts=True)
            log.info(f"Assigned {counts.tolist()} breached embeddings to each sentence.")
        else:
            sentence_labels = torch.zeros(len(breached_embeddings), dtype=torch.long)

        # Match breached embeddings to positions for each sentence:
        breached_embeddings = breached_embeddings[:, v_length:-1]
        positional_embeddings = positional_embeddings[:, v_length:-1]
        leaked_embeddings = leaked_embeddings[:, v_length:-1]

        # First assign and remove the token id from each breached embedding
        order_leaked_to_breached, costs = self._match_embeddings(leaked_embeddings, breached_embeddings)
        recovered_tokens = leaked_tokens[order_leaked_to_breached]
        if self.cfg.embedding_token_weight > 0:
            # Optionally: Match breached_without_positions to any embedding entries
            # If the costs from the matching above are low, then this can recover lost tokens that were missed by
            # .recover_token_information()
            vocab_size = server_payload[0]["metadata"]["vocab_size"]
            all_token_ids = torch.arange(0, vocab_size, device=self.setup["device"])
            all_token_embeddings = lookup["norm_layer1"](lookup["embedding"](all_token_ids))
            all_token_embeddings = all_token_embeddings[:, v_length:-1]
            for idx, entry in enumerate(breached_embeddings.detach().cpu().numpy()):
                max_corr = self.vcorrcoef(all_token_embeddings.detach()[1:].cpu().numpy(), entry)
                val, loc = torch.as_tensor(max_corr).max(dim=0)
                if val * self.cfg.embedding_token_weight > costs[idx]:
                    log.info(f"Replaced token {idx} with corr {costs[idx]} with new token with corr {val}")
                    recovered_tokens[idx] = loc + 1
        breached_token_embeddings = lookup["norm_layer1"](
            lookup["embedding"](recovered_tokens.to(device=self.setup["device"]))
        )[:, v_length:-1]
        breached_just_positions = breached_embeddings - breached_token_embeddings
        ordered_tokens = -torch.ones(len_data * data_shape[0], dtype=torch.long)
        for sentence in range(len_data):
            order_breach_to_positions, costs = self._match_embeddings(
                positional_embeddings[: data_shape[0]], breached_just_positions[sentence_labels == sentence]
            )
            ordered_tokens[sentence * data_shape[0] + order_breach_to_positions] = recovered_tokens[
                sentence_labels == sentence
            ]
        # Then fill up the missing locations:
        if len(breached_just_positions) < len(positional_embeddings):
            free_positions = (ordered_tokens == -1).nonzero().squeeze(dim=1)
            assert len(breached_just_positions) > len(free_positions)
            miss_to_pos, costs = self._match_embeddings(breached_just_positions, positional_embeddings[free_positions])
            ordered_tokens[free_positions] = recovered_tokens[miss_to_pos]

        # Finally re-order into sentences:
        final_tokens = ordered_tokens.view([len_data, *data_shape])
        reconstructed_data = dict(data=final_tokens, labels=final_tokens)
        return reconstructed_data, stats

    @torch.inference_mode()
    def reconstruct_single_sentence(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """Reconstruct both positions and token ids from the input sentence.

        Old method for sanity checks. Cannot do sentence disambiguation or understand v_length."""
        # Initialize stats module for later usage:
        rec_models, tokens, stats = self.prepare_attack(server_payload, shared_data)
        len_data = shared_data[0]["metadata"]["num_data_points"]  # Could be guessed as well

        if "transformer" in rec_models[0].name:  # These are our implementations from model/language_models.py
            norm_layer = rec_models[0].transformer_encoder.layers[0].norm1
            pos_encoder = rec_models[0].pos_encoder
            embedding = rec_models[0].encoder

        if "ImprintBlock" in server_secrets.keys():
            weight_idx = server_secrets["ImprintBlock"]["weight_idx"]
            bias_idx = server_secrets["ImprintBlock"]["bias_idx"]
            data_shape = server_secrets["ImprintBlock"]["data_shape"]
        else:
            raise ValueError(f"No imprint hidden in model {rec_models[0]} according to server.")

        leaked_tokens = self.recover_token_information(shared_data, server_payload, rec_models).view(-1)
        leaked_embeddings = norm_layer(embedding(leaked_tokens)).cpu().view(-1, embedding.weight.shape[1])

        bias_grad = torch.cat([shared_data[0]["gradients"][b_idx].clone() for b_idx in bias_idx])
        weight_grad = torch.cat([shared_data[0]["gradients"][w_idx].clone() for w_idx in weight_idx])

        if self.cfg.sort_by_bias:
            # This variant can recover from shuffled rows under the assumption that biases would be ordered
            _, order = server_payload[0]["parameters"][1].sort(descending=True)
            bias_grad = bias_grad[order]
            weight_grad = weight_grad[order]

        if server_secrets["ImprintBlock"]["structure"] == "cumulative":
            for i in reversed(list(range(1, weight_grad.shape[0]))):
                weight_grad[i] -= weight_grad[i - 1]
                bias_grad[i] -= bias_grad[i - 1]

        # Here are our reconstructed positionally encoded embeddings:
        valid_classes = bias_grad != 0
        breached_embeddings = weight_grad[valid_classes, :] / bias_grad[valid_classes, None]
        log.info(f"Recovered {len(breached_embeddings)} embeddings with positional data from imprinted layer.")
        # Get an estimation of the positional embeddings:
        dummy_inputs = torch.zeros([len_data, *data_shape], dtype=torch.long, device=self.setup["device"])
        pure_positions = pos_encoder(torch.zeros_like(embedding(dummy_inputs)))
        positional_embeddings = norm_layer(pure_positions).cpu().view(-1, embedding.weight.shape[1])
        # First, match breached embeddings to positions:
        order_breach_to_positions, costs = self._match_embeddings(positional_embeddings, breached_embeddings)
        if len(order_breach_to_positions) < len(positional_embeddings):
            # First fill up found positions:
            ordered_breached_embeddings = torch.zeros_like(positional_embeddings)
            ordered_breached_embeddings[order_breach_to_positions] = breached_embeddings
            # Then dredge for other hits from rows that do not correlate well
            positions_to_breach = torch.arange(0, len(positional_embeddings))[order_breach_to_positions]
            free_positions = list(set(range(0, len(positional_embeddings))) - set(positions_to_breach.tolist()))
            # _, indices = costs.topk(len(free_positions), largest=False)
            miss_to_pos, costs = self._match_embeddings(breached_embeddings, positional_embeddings[free_positions])
            ordered_breached_embeddings[free_positions] = breached_embeddings[miss_to_pos]
            # ordered_breached_embeddings[free_positions] -= breached_embeddings[indices]  # remove overlapping positions
        else:  # Match 1-to-1:
            ordered_breached_embeddings = breached_embeddings[order_breach_to_positions]

        # These are already in the right position, but which token do they belong to?
        breached_without_positions = ordered_breached_embeddings - positional_embeddings
        order_leaked_to_breached, _ = self._match_embeddings(leaked_embeddings, breached_without_positions)
        recovered_tokens = leaked_tokens[order_leaked_to_breached].view([len_data, *data_shape])

        reconstructed_data = dict(data=recovered_tokens, labels=recovered_tokens)
        return reconstructed_data, stats

    def _match_frequency_estimation(self, positional_embeddings, breached_embeddings):
        """Match multiple rows to single positions, based on their norm."""
        breaches = breached_embeddings.clone()
        matches = []
        while len(matches) < len(positional_embeddings):
            bnorm = breaches.norm(dim=-1)[:, None]
            pnorm = positional_embeddings.norm(dim=-1)[None, :]
            cosim = breaches.matmul(positional_embeddings.T) / bnorm / pnorm
            row_ind, col_ind = (cosim == torch.max(cosim)).nonzero()[0].squeeze()

            matches.append(row_ind)
            breaches[row_ind] -= positional_embeddings[col_ind]

        order_tensor = torch.as_tensor(matches, device=positional_embeddings.device, dtype=torch.long)
        return order_tensor, None

    def _match_embeddings(self, inputs, references, measure="corrcoef", fallbacks=None):
        if references.ndim == 1:
            references = references[None, :]
        if measure == "corrcoef":
            s, e = inputs.shape
            corr = np.corrcoef(inputs.detach().cpu().numpy(), references.detach().cpu().numpy())[s:, :s]
        else:
            corr = references.matmul(inputs.T) / references.norm(dim=-1)[:, None] / inputs.norm(dim=-1)[None, :]
            corr = corr.detach().numpy()
        try:
            row_ind, col_ind = linear_sum_assignment(corr, maximize=True)
        except ValueError:
            log.info(f"ValueError from correlation matrix {corr}")
            if fallbacks is None:
                log.info("Returning trivial order...")
                row_ind, col_ind = list(range(corr.shape[1])), list(range(corr.shape[0]))
            else:
                log.info("Returning fallback order...")
                row_ind, col_ind = list(range(corr.shape[0])), fallbacks
        order_tensor = torch.as_tensor(col_ind, device=inputs.device, dtype=torch.long)
        costs = torch.as_tensor(corr[row_ind, col_ind], device=inputs.device, dtype=torch.float)
        return order_tensor, costs

    def reconstruct2(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """Reconstruct both positions and token ids from the input sentence.

        Oldest implementation with padding added on top."""
        # Initialize stats module for later usage:
        rec_models, tokens, stats = self.prepare_attack(server_payload, shared_data)
        len_data = shared_data[0]["metadata"]["num_data_points"]  # Could be guessed as well

        if "ImprintBlock" in server_secrets.keys():
            weight_idx = server_secrets["ImprintBlock"]["weight_idx"]
            bias_idx = server_secrets["ImprintBlock"]["bias_idx"]
            data_shape = server_secrets["ImprintBlock"]["data_shape"]
        else:
            raise ValueError(f"No imprint hidden in model {rec_models[0]} according to server.")

        leaked_tokens = self.recover_token_information(shared_data, server_payload, rec_models).unique().view(-1)

        bias_grad = torch.cat([shared_data[0]["gradients"][b_idx].clone() for b_idx in bias_idx])
        weight_grad = torch.cat([shared_data[0]["gradients"][w_idx].clone() for w_idx in weight_idx])

        if self.cfg.sort_by_bias:
            # This variant can recover from shuffled rows under the assumption that biases would be ordered
            _, order = server_payload[0]["parameters"][1].sort(descending=True)
            bias_grad = bias_grad[order]
            weight_grad = weight_grad[order]

        if server_secrets["ImprintBlock"]["structure"] == "cumulative":
            for i in reversed(list(range(1, weight_grad.shape[0]))):
                weight_grad[i] -= weight_grad[i - 1]
                bias_grad[i] -= bias_grad[i - 1]

        # Here are our reconstructed positionally encoded features:
        valid_classes = bias_grad != 0
        recs = weight_grad[valid_classes, :] / bias_grad[valid_classes, None]

        # Associate tokens with embeddings:
        # First, let's get the features of our bag of words sans positional encoding
        if "transformer" in rec_models[0].name:  # These are our implementations from model/language_models.py
            norm_layer = rec_models[0].transformer_encoder.layers[0].norm1
            pos_encoder = rec_models[0].pos_encoder
            embedding = rec_models[0].encoder
        no_pos = norm_layer(embedding(leaked_tokens)).cpu().view(-1, embedding.weight.shape[1])
        with_pos = recs  # Here are those same features, but with positional encodings (stuff we reconstructed)
        log.info(f"Recovered {len(with_pos)} tokens with positional data from imprinted layer.")
        # if with_pos.shape[0] < len_data:
        #     # Some positional tokens are missing, we substitute with random vectors:
        #     random_embeds = torch.zeros(len_data - with_pos.shape[0], with_pos.shape[1])
        #     with_pos = torch.cat([with_pos, random_embeds])

        sorted_tokens, order = self._match_positions_with_tokens(no_pos, with_pos, leaked_tokens)
        # ### Now let's get each token's position, as well as splitting sequences
        # Now that we've 'lined-up' the pos-encoded features with non-pos-encoded features, let's subtract the two
        # to get some 'faux' positions (layer norm means they aren't exact).
        estimated_pos = torch.stack([with_pos[order[i][1]] - no_pos[order[i][0]] for i in range(len(order))])
        new_with_pos = [with_pos[order[i][1]] for i in range(len(order))]

        # Now let's get just the additive part of the positional encoding
        dummy_inputs = torch.zeros([len_data, *data_shape], dtype=torch.long, device=self.setup["device"])
        just_pos = norm_layer(pos_encoder(torch.zeros_like(embedding(dummy_inputs)))).cpu().squeeze()

        # Getting multiple user's sentences back
        sentences = self._match_words_to_sentences(estimated_pos, just_pos, new_with_pos, len_data, sorted_tokens)

        # Pad recovered sentences:
        PAD_token = 0
        final_tokens = torch.ones([len_data, *data_shape], dtype=torch.long) * PAD_token
        for idx, sentence in enumerate(sentences):
            for widx, word in enumerate(sentence):
                final_tokens[idx, widx] = word

        reconstructed_data = dict(data=final_tokens, labels=final_tokens)
        return reconstructed_data, stats

    def _match_positions_with_tokens(self, no_pos, with_pos, leaked_tokens):
        corrs = torch.zeros((len(no_pos), len(with_pos)))

        # ### We need to find out what word led to what positionally encoded representation.
        # Let's try the naive greedy search for correlations between no_pos and with_pos as defined above
        for i, no_p in enumerate(no_pos):
            for j, with_p in enumerate(with_pos):
                val = np.corrcoef(np.array([no_p.detach().numpy(), with_p.detach().numpy()]))[0, 1]
                corrs[i, j] = val

        # Find which positionally-encoded vector associates with un-positionally-encoded vector
        row_ind, col_ind = linear_sum_assignment(corrs.numpy(), maximize=True)

        order = [(row_i, col_i) for (row_i, col_i) in zip(row_ind, col_ind)]
        order = sorted(order, key=lambda x: x[1])

        # Now let's re-sort the tokens by this order
        sorted_tokens = [leaked_tokens[order[i][0]] for i in range(len(order))]
        return sorted_tokens, order

    def _match_words_to_sentences(self, estimated_pos, just_pos, new_with_pos, len_data, sorted_tokens):
        # Let's calculate this matrix again, but for the new method (previous calculation was just for old method, can ignore)
        order_coeffs = torch.zeros((len(estimated_pos), len(just_pos)))
        for i in range(len(estimated_pos)):
            for j in range(len(just_pos)):
                order_coeffs[i, j] = np.corrcoef(estimated_pos[i].detach().numpy(), just_pos[j].detach().numpy())[0, 1]

        # Now, we make a dictionary where keys are positions, and values are encoded embeddings.
        # i.e. word_groups[0] = ['0th_word_of_sequence1', '0th_word_of_sequence2', ...]

        word_groups = defaultdict(list)

        for i in range(order_coeffs.shape[0]):
            max_corr = torch.argmax(order_coeffs[i]).item()
            word_groups[max_corr].append(i)

        # Sort these word groups to start forming sentences
        sorted_keys = sorted([k for k in word_groups.keys()])
        word_groups = [word_groups[k] for k in sorted_keys]
        first_words = word_groups[0]

        sentences = [[] for i in range(len_data)]

        # Start the sentences with first words
        for i, first_w in enumerate(first_words):
            sentences[i].append(sorted_tokens[first_w])

        # Go through the rest of the word groups, assigning words to their appropriate sentences
        for w in word_groups[1:]:
            corr = torch.zeros(len(w), len(first_words))
            for i, x in enumerate(w):
                for j, y in enumerate(first_words):
                    corr[i, j] = np.corrcoef(estimated_pos[x].detach().numpy(), new_with_pos[y].detach().numpy())[0, 1]

            # Below we do linear sum assignment for each word to each potential sentence
            row_ind, col_ind = linear_sum_assignment(corr.numpy(), maximize=True)
            for m, n in zip(row_ind, col_ind):
                sentences[n].append(sorted_tokens[w[m]])

        return sentences

    @staticmethod
    def vcorrcoef(X, y):
        """Correlation between matrix and vector taken from here because lazy:
        https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/
        """
        Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
        ym = np.mean(y)
        r_num = np.sum((X - Xm) * (y - ym), axis=1)
        r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
        r = r_num / r_den
        return r


class AprilAttacker(AnalyticAttacker):
    """Analytically reconstruct the input of a vision transformer for a batch of size 1."""

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """Analytic recovery based on first query."""
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        len_data = shared_data[0]["metadata"]["num_data_points"]  # Not strictly needed for the attack

        x_patched = self.closed_form_april(rec_models[0], shared_data[0]).to(**self.setup)
        x = self.recover_patch(x_patched)
        inputs = torch.max(torch.min(x, (1 - self.dm) / self.ds), -self.dm / self.ds)

        # This attack always recovers just one sample!
        data = torch.zeros([len_data, *self.data_shape], **self.setup)
        data[0] = inputs
        reconstructed_data = dict(data=data, labels=labels)
        return reconstructed_data, stats

    @staticmethod
    def recover_patch(x):
        # retile img
        p_num_2, p_size_2 = x.shape[1:]
        p_num = int(p_num_2 ** (1 / 2))
        p_size = int(p_size_2 ** (1 / 2))
        img_size = int((p_num_2 * p_size_2) ** (1 / 2))
        x = x.reshape((3, p_num, p_num, p_size, p_size))
        new_x = torch.zeros_like(x).reshape((3, img_size, img_size))

        for i in range(p_num):
            for j in range(p_num):
                new_x[:, i * p_size : (i + 1) * p_size, j * p_size : (j + 1) * p_size] = x[:, i, j, :]

        return new_x

    @staticmethod
    def closed_form_april(model, shared_data):
        """Run inversions on CPU in double precision. (gelsd only implemented for CPU)"""
        # recover patch embeddings first, (APRIL paper)
        qkv_w = model.model.blocks[0].attn.qkv.weight.detach().double().cpu()
        q_w, k_w, v_w = qkv_w.reshape(3, -1, qkv_w.shape[-1]).unbind()
        qkv_g = shared_data["gradients"][4].detach().double().cpu()
        assert qkv_w.shape == qkv_g.shape
        q_g, k_g, v_g = qkv_g.reshape(3, -1, qkv_g.shape[-1]).unbind()
        A = shared_data["gradients"][1].detach().squeeze().double().cpu()
        pos_embed = model.model.pos_embed.detach().squeeze().double().cpu()

        b = (q_w.T @ q_g + k_w.T @ k_g + v_w.T @ v_g).double().cpu()
        log.info(f"Attention Inversion:  ||A||={A.norm()}, ||b||={b.norm()}")
        z = torch.linalg.lstsq(A @ A.T, A @ b, driver="gelsd", rcond=None).solution
        z -= pos_embed
        z = z[1:]

        # recover img
        em_w = model.model.patch_embed.proj.weight.detach().double().cpu()
        em_w = em_w.reshape((em_w.shape[0], -1))
        em_b = model.model.patch_embed.proj.bias.detach().double().cpu()

        x = z - em_b
        log.info(f"Embedding Inversion:  ||A||={em_w.norm()}, ||b||={x.norm()}")
        x = torch.linalg.lstsq(em_w, x.T, driver="gelsd", rcond=None).solution
        x = x.reshape((3, -1, x.shape[-1]))
        x = x.transpose(1, 2)
        return x
