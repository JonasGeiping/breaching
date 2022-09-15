"""A variety of analytic attacks. Which, to be more precise more-or-less means "non"-iterative attacks,
in differentiation from the optimization-based attacks."""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment  # Better than greedy search


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
            for layer in list(model.modules())[::-1]:  # Only for torch.nn.Sequential
                if isinstance(layer, torch.nn.Linear):
                    bias_grad = user_data["gradients"][idx]
                    weight_grad = user_data["gradients"][idx - 1]
                    layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, labels)
                    idx -= 2
                elif isinstance(layer, torch.nn.Flatten):
                    inputs = layer_inputs.reshape(user_data["metadata"]["num_data_points"], *self.data_shape)
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

        if "ImprintBlock" in server_secrets.keys():
            weight_idx = server_secrets["ImprintBlock"]["weight_idx"]
            bias_idx = server_secrets["ImprintBlock"]["bias_idx"]
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

        # This is the attack:
        layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, [])

        # Reduce hits if necessary:
        layer_inputs = self.reduce_hits(layer_inputs, weight_grad, bias_grad, shared_data)

        # Reshape images, re-identify token embeddings:
        reconstructed_inputs = self.reformat_data(layer_inputs, rec_models, shared_data, server_payload, server_secrets)
        reconstructed_user_data = dict(data=reconstructed_inputs, labels=labels)

        return reconstructed_user_data, stats

    def reduce_hits(self, layer_inputs, weight_grad, bias_grad, shared_data):
        """In case of numerical instability or gradient noise, more bins can be hit than expected."""
        log.info(f"Initially produced {layer_inputs.shape[0]} hits.")
        len_data = shared_data[0]["metadata"]["num_data_points"]  # Not strictly needed for the attack, used to pad/trim
        if len_data >= layer_inputs.shape[0]:
            # Fill up with zero if not enough data can be found?
            if self.cfg.breach_padding:
                missing_entries = layer_inputs.new_zeros(len_data - layer_inputs.shape[0], *layer_inputs.shape[1:])
                layer_inputs = torch.cat([layer_inputs, missing_entries], dim=0)
        else:
            # Cut additional hits:
            if self.cfg.breach_reduction == "bias":
                # this rule is optimal for clean data with few bins:
                best_guesses = torch.topk(bias_grad[bias_grad != 0].abs(), len_data, largest=False)[1]
            elif self.cfg.breach_reduction == "weight":
                # this rule is best when faced with differential privacy:
                best_guesses = torch.topk(weight_grad.mean(dim=1)[bias_grad != 0].abs(), len_data, largest=False)[1]
            else:  # None #
                # Warning: This option can mess up metrics later on (as more data is recpnstructed than exists)
                best_guesses = torch.arange(layer_inputs.shape[0])
            if len(best_guesses) < len_data:
                log.info(f"Reduced to {len_data} hits.")
            layer_inputs = layer_inputs[best_guesses]
        return layer_inputs

    def reformat_data(self, layer_inputs, rec_models, shared_data, server_payload, server_secrets):
        """After the actual attack has happened, we have to do some work to piece everything back together in the
        desired data format."""

        data_shape = server_secrets["ImprintBlock"]["shape"]

        if "decoder" in server_secrets["ImprintBlock"].keys():
            inputs = server_secrets["ImprintBlock"]["decoder"](layer_inputs)

        if server_payload[0]["metadata"].modality == "vision":
            data_dtype = self.setup["dtype"]
            inputs = layer_inputs.reshape(layer_inputs.shape[0], *data_shape)[:, :3, :, :]
            if inputs.shape[1:] != self.data_shape:
                interp_mode = dict(mode="bicubic", align_corners=False)
                inputs = torch.nn.functional.interpolate(inputs, size=self.data_shape[1:], **interp_mode)
            inputs = torch.max(torch.min(inputs, (1 - self.dm) / self.ds), -self.dm / self.ds)
        else:
            data_dtype = torch.long
            inputs = layer_inputs.reshape(layer_inputs.shape[0], *data_shape)
            if self.cfg.token_strategy is not None:
                leaked_tokens = self.recover_token_information(shared_data, server_payload, rec_models[0].name)
            inputs = self._postprocess_text_data(dict(data=inputs, labels=leaked_tokens), models=rec_models)["data"]

        return inputs


class DecepticonAttacker(AnalyticAttacker):
    """An analytic attack against transformer models in language."""

    @torch.inference_mode()
    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """Reconstruct both positions and token ids from the input sentence. Disambiguate sentences based on v_length."""
        # Initialize stats module for later usage:
        rec_models, _, stats = self.prepare_attack(server_payload, shared_data)
        len_data = shared_data[0]["metadata"]["num_data_points"]  # Could be guessed as well
        lookup = lookup_module_names(rec_models[0].name, rec_models[0])

        if "ImprintBlock" in server_secrets.keys():
            data_shape = server_secrets["ImprintBlock"]["data_shape"]
            v_length = server_secrets["ImprintBlock"]["v_length"]
        else:
            raise ValueError(f"No imprint hidden in model {rec_models[0]} according to server.")
        [model.eval() for model in rec_models]

        # Estimate all tokens
        leaked_tokens = self.recover_token_information(shared_data, server_payload, rec_models[0].name)
        if leaked_tokens is not None:
            leaked_embeddings = lookup["norm_layer1"](lookup["embedding"](leaked_tokens.view(-1)))
            leaked_embeddings = leaked_embeddings.cpu().view(-1, lookup["embedding"].weight.shape[1])

        # Extract embeddings from linear layers
        breached_embeddings = self._extract_breaches(shared_data, server_payload, server_secrets)

        # Get an estimation of the positional embeddings:
        dummy_inputs = torch.zeros([len_data, *data_shape], dtype=torch.long, device=self.setup["device"])
        pure_positions = lookup["pos_encoder"](torch.zeros_like(lookup["embedding"](dummy_inputs)))
        pure_normed_positions = lookup["norm_layer1"](pure_positions)
        positional_embeddings = pure_normed_positions.cpu().view(-1, lookup["embedding"].weight.shape[1])
        positional_embeddings = positional_embeddings.to(dtype=self.setup["dtype"])

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

        # Sentence-based backfill:
        if self.cfg.sentence_based_backfill:
            breached_embeddings, sentence_labels = self._sentence_backfill(
                breached_embeddings, sentence_labels, [len_data, data_shape[0]], v_length
            )

        # Match breached embeddings to positions for each sentence:
        breached_embeddings = breached_embeddings[:, v_length:-1]
        positional_embeddings = positional_embeddings[:, v_length:-1]
        leaked_embeddings = leaked_embeddings[:, v_length:-1] if leaked_tokens is not None else None

        if self.cfg.recovery_order == "positions-first":
            # First assign and remove the position from each breached embedding
            ordered_breached_embeddings = torch.zeros_like(positional_embeddings)
            for sentence in range(len_data):
                order_breach_to_positions, _, costs = self._match_embeddings(
                    positional_embeddings[: data_shape[0]],
                    breached_embeddings[sentence_labels == sentence],
                    measure=self.cfg.matcher,
                )
                ordered_breached_embeddings[sentence * data_shape[0] + order_breach_to_positions] = breached_embeddings[
                    sentence_labels == sentence
                ]
                if self.cfg.backfill_removal is not None:
                    # Remove filled position
                    breached_embeddings[sentence_labels == sentence] = self._separate(
                        breached_embeddings[sentence_labels == sentence],
                        positional_embeddings[: data_shape[0]][order_breach_to_positions],
                    )

            if len(breached_embeddings) < len(positional_embeddings):
                # Then fill up the missing locations:
                ordered_breached_embeddings = self._backfill_embeddings(
                    ordered_breached_embeddings, breached_embeddings, positional_embeddings, sentence_labels, data_shape
                )

            # These are already in the right position, but which token do they belong to?
            breached_without_positions = self._separate(ordered_breached_embeddings, positional_embeddings)
            if leaked_tokens is not None:
                order_leaked_to_breached, _, costs = self._match_embeddings(
                    leaked_embeddings,
                    breached_without_positions,
                    measure=self.cfg.matcher,
                )
                recovered_tokens = leaked_tokens.view(-1)[order_leaked_to_breached]
            else:
                recovered_tokens = torch.zeros(len_data * data_shape[0], dtype=torch.long)
                costs = -float("inf") * torch.ones(len_data * data_shape[0])
            if self.cfg.embedding_token_weight > 0 or leaked_tokens is None:
                recovered_tokens = self._supplement_from_full_vocabulary(
                    recovered_tokens, costs, breached_without_positions, v_length, lookup
                )

            # Finally reshape into sentences:
            final_tokens = recovered_tokens.view([len_data, *data_shape])

        elif self.cfg.recovery_order == "tokens-first":
            # First assign and remove the token id from each breached embedding
            if leaked_tokens is not None:
                order_leaked_to_breached, _, costs = self._match_embeddings(
                    leaked_embeddings, breached_embeddings, measure=self.cfg.matcher
                )
                recovered_tokens = leaked_tokens.view(-1)[order_leaked_to_breached]
            else:
                recovered_tokens = torch.zeros(len(breached_embeddings), dtype=torch.long)
                costs = -float("inf") * torch.ones(len(breached_embeddings))
            if self.cfg.embedding_token_weight > 0 or leaked_tokens is None:
                recovered_tokens = self._supplement_from_full_vocabulary(
                    recovered_tokens, costs, breached_embeddings, v_length, lookup
                )
            breached_token_embeddings = (
                lookup["norm_layer1"](lookup["embedding"](recovered_tokens.to(device=self.setup["device"])))[
                    :, v_length:-1
                ]
                .cpu()
                .to(dtype=self.setup["dtype"])
            )
            breached_just_positions = self._separate(breached_embeddings, breached_token_embeddings)
            ordered_tokens = -torch.ones(len_data * data_shape[0], dtype=torch.long)

            for sentence in range(len_data):
                order_breach_to_positions, _, costs = self._match_embeddings(
                    positional_embeddings[: data_shape[0]],
                    breached_just_positions[sentence_labels == sentence],
                    measure=self.cfg.matcher,
                )
                ordered_tokens[sentence * data_shape[0] + order_breach_to_positions] = recovered_tokens[
                    sentence_labels == sentence
                ]
                if self.cfg.backfill_removal is not None:
                    # Remove filled position
                    breached_embeddings[sentence_labels == sentence] = self._separate(
                        breached_embeddings[sentence_labels == sentence],
                        positional_embeddings[: data_shape[0]][order_breach_to_positions],
                    )
                    # # Remove filled token
                    # breached_embeddings[sentence_labels == sentence] = self._separate(
                    #     breached_embeddings[sentence_labels == sentence],
                    #     breached_token_embeddings[sentence_labels == sentence],
                    # )
            # Then fill up the missing locations:
            if len(breached_embeddings) < len(positional_embeddings):
                ordered_tokens = self._backfill_tokens(
                    ordered_tokens,
                    breached_embeddings,
                    positional_embeddings,
                    sentence_labels,
                    data_shape,
                    recovered_tokens=recovered_tokens,
                )

            # Finally reshape into sentences:
            final_tokens = ordered_tokens.view([len_data, *data_shape])

        else:
            raise ValueError(f"Invalid recovery order {self.cfg.recovery_order} given.")

        confidence_per_token = self._compute_confidence_estimates(final_tokens, breached_embeddings, v_length, lookup)

        reconstructed_data = dict(data=final_tokens, labels=final_tokens, confidence=confidence_per_token)
        return reconstructed_data, stats

    def _extract_breaches(self, shared_data, server_payload, server_secrets):
        """Extract breached embeddings from linear layers. Handles some of the ugly complexity like
        * Transposing for conv-type implementations
        * Resorting biases if they were unsorted (to hide the attack)
        * Invert cumulative sum structure
        * --- Do the actual extraction by weight_grad divided by bias_grad ---
        * Cast to correct data types
        * Remove extraneous hits (for example because of gradient noise)
        * Remove NaNs if any
        """
        weight_idx = server_secrets["ImprintBlock"]["weight_idx"]
        bias_idx = server_secrets["ImprintBlock"]["bias_idx"]
        bias_grad = torch.cat([shared_data[0]["gradients"][b_idx].clone() for b_idx in bias_idx])
        if server_secrets["ImprintBlock"]["ff_transposed"]:
            weight_grad = torch.cat([shared_data[0]["gradients"][w_idx].clone() for w_idx in weight_idx], dim=1)
            weight_grad = weight_grad.T.contiguous()  # Restride only due to paranoia
        else:
            weight_grad = torch.cat([shared_data[0]["gradients"][w_idx].clone() for w_idx in weight_idx])
        weight_grad = weight_grad.to(dtype=self.setup["dtype"])  # up-cast if necessary
        bias_grad = bias_grad.to(dtype=self.setup["dtype"])  # up-cast if necessary

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
        if self.cfg.undivided:
            breached_embeddings = weight_grad[valid_classes, :]
            std, mean = torch.std_mean(breached_embeddings)
            breached_embeddings = (breached_embeddings - mean) / (std + 1e-8)
        else:
            breached_embeddings = weight_grad[valid_classes, :] / bias_grad[valid_classes, None]
        not_nan_positions = ~torch.isnan(breached_embeddings.sum(dim=-1))  # This should usually be all positions
        breached_embeddings = breached_embeddings[not_nan_positions]  # just being paranoid here
        log.info(f"Recovered {len(breached_embeddings)} embeddings with positional data from imprinted layer.")

        # Sometimes too many rows activate (more than expected data), due to gradient noise or numerical issues.
        # In that case only a subset of most-likely-to-be-real embeddings should be used
        len_data = shared_data[0]["metadata"]["num_data_points"]
        data_shape = server_secrets["ImprintBlock"]["data_shape"]
        # print(weight_grad.mean(dim=1).topk(k=64)[0])
        # print(weight_grad.abs().mean(dim=1).topk(k=64)[0])
        if len(breached_embeddings) > len_data * data_shape[0]:
            if self.cfg.breach_reduction == "weight":
                best_guesses = torch.topk(
                    weight_grad.mean(dim=1)[bias_grad != 0].abs(), len_data * data_shape[0], largest=True
                )
            elif self.cfg.breach_reduction == "total-weight":
                best_guesses = torch.topk(
                    weight_grad.pow(2).sum(dim=1)[bias_grad != 0], len_data * data_shape[0], largest=True
                )

            elif self.cfg.breach_reduction == "bias":
                best_guesses = torch.topk(bias_grad[bias_grad != 0].abs(), len_data * data_shape[0], largest=False)
            else:
                raise ValueError(f"Invalid breach reduction {self.cfg.breach_reduction} given.")
            log.info(f"Reduced to {len_data * data_shape[0]} hits.")
            breached_embeddings = breached_embeddings[best_guesses.indices]

        # Cast to CPU
        breached_embeddings = breached_embeddings.cpu().to(dtype=self.setup["dtype"])  # Assignments run on CPU anyway

        # Final assertion of sensible values
        if (~torch.isfinite(breached_embeddings)).sum():
            raise ValueError("Invalid breached embeddings recovered.")
        return breached_embeddings

    def _backfill_embeddings(
        self,
        ordered_embeddings,
        fillable_embeddings,
        positional_embeddings,
        sentence_labels,
        data_shape,
    ):
        """Fill missing positions in ordered_embeddings based on some heuristic
        with collisions from fillable_embeddings.
        This method has a good amount of overlap with _backfill_tokens but combining them was just a mess of
        if inputs_are_tokens, then ...
        """
        free_positions = (ordered_embeddings.norm(dim=-1) == 0).nonzero().squeeze(dim=1)

        if self.cfg.backfilling == "global":
            # Fill missing locations globally
            while len(free_positions) > 0:
                order_breach_to_positions, selection_tensor, costs = self._match_embeddings(
                    positional_embeddings[free_positions], fillable_embeddings, measure=self.cfg.matcher
                )
                ordered_embeddings[free_positions[order_breach_to_positions]] = fillable_embeddings[selection_tensor]
                if self.cfg.backfill_removal is not None:
                    fillable_embeddings[selection_tensor] = self._separate(
                        fillable_embeddings[selection_tensor],
                        positional_embeddings[free_positions][order_breach_to_positions],
                    )
                free_positions = (ordered_embeddings.norm(dim=-1) == 0).nonzero().squeeze(dim=1)

        elif self.cfg.backfilling == "local":
            # Fill sentence-by-sentence
            for sentence in range(data_shape[0]):
                sentence_inputs = ordered_embeddings[sentence * data_shape[0] : (sentence + 1) * data_shape[0]]
                free_positions = (sentence_inputs.norm(dim=-1) == 0).nonzero().squeeze(dim=1)
                while len(free_positions) > 0:
                    order_breach_to_positions, selection_tensor, costs = self._match_embeddings(
                        positional_embeddings[: data_shape[0]][free_positions],
                        fillable_embeddings[sentence_labels == sentence],
                        measure=self.cfg.matcher,
                    )
                    sentence_inputs[free_positions[order_breach_to_positions]] = fillable_embeddings[
                        sentence_labels == sentence
                    ][selection_tensor]
                    if self.cfg.backfill_removal is not None:
                        fillable_embeddings[sentence_labels == sentence][selection_tensor] = self._separate(
                            fillable_embeddings[sentence_labels == sentence][selection_tensor],
                            positional_embeddings[: data_shape[0]][free_positions][order_breach_to_positions],
                        )
                    free_positions = (sentence_inputs.norm(dim=-1) == 0).nonzero().squeeze(dim=1)
                ordered_embeddings[sentence * data_shape[0] : (sentence + 1) * data_shape[0]] = sentence_inputs

        elif self.cfg.backfilling == "randn":  # sanity check option
            ordered_embeddings[free_positions] = torch.randn(
                [len(free_positions), ordered_embeddings.shape[-1]], dtype=self.setup["dtype"]
            )
        else:
            raise ValueError(f"Invalid backfilling heuristic {self.cfg.backfilling} given.")

        return ordered_embeddings

    def _backfill_tokens(
        self,
        ordered_tokens,
        fillable_embeddings,
        positional_embeddings,
        sentence_labels,
        data_shape,
        recovered_tokens=None,
    ):
        """Fill missing positions in ordered_tokens based on some heuristic
        with collisions from fillable_embeddings.
        recovered_tokens has to be a lookup for the tokens corresponding to fillable_embeddings
        This method has a good amount of overlap with _backfill_embeddings but combining them was just a mess of
        if inputs_are_tokens, then ...
        """
        free_positions = (ordered_tokens == -1).nonzero().squeeze(dim=1)

        if self.cfg.backfilling == "global":
            # Fill missing locations globally
            while len(free_positions) > 0:
                order_breach_to_positions, selection_tensor, costs = self._match_embeddings(
                    positional_embeddings[free_positions], fillable_embeddings, measure=self.cfg.matcher
                )
                ordered_tokens[free_positions[order_breach_to_positions]] = recovered_tokens[selection_tensor]
                if self.cfg.backfill_removal is not None:
                    fillable_embeddings[selection_tensor] = self._separate(
                        fillable_embeddings[selection_tensor],
                        positional_embeddings[free_positions][order_breach_to_positions],
                    )
                free_positions = (ordered_tokens == -1).nonzero().squeeze(dim=1)

        elif self.cfg.backfilling == "local":
            # Fill sentence-by-sentence
            for sentence in range(data_shape[0]):
                sentence_inputs = ordered_tokens[sentence * data_shape[0] : (sentence + 1) * data_shape[0]]
                free_positions = (sentence_inputs == -1).nonzero().squeeze(dim=1)
                while len(free_positions) > 0:
                    order_breach_to_positions, selection_tensor, costs = self._match_embeddings(
                        positional_embeddings[: data_shape[0]][free_positions],
                        fillable_embeddings[sentence_labels == sentence],
                        measure=self.cfg.matcher,
                    )
                    sentence_inputs[free_positions[order_breach_to_positions]] = recovered_tokens[
                        sentence_labels == sentence
                    ][selection_tensor]
                    if self.cfg.backfill_removal is not None:
                        fillable_embeddings[sentence_labels == sentence][selection_tensor] = self._separate(
                            fillable_embeddings[sentence_labels == sentence][selection_tensor],
                            positional_embeddings[: data_shape[0]][free_positions][order_breach_to_positions],
                        )
                    free_positions = (sentence_inputs == -1).nonzero().squeeze(dim=1)
                ordered_tokens[sentence * data_shape[0] : (sentence + 1) * data_shape[0]] = sentence_inputs

        elif self.cfg.backfilling == "randn":  # sanity check option
            ordered_tokens[free_positions] = torch.randint(
                0, ordered_tokens.max(), (len(free_positions),), dtype=torch.long
            )
        else:
            raise ValueError(f"Invalid backfilling heuristic {self.cfg.backfilling} given.")

        return ordered_tokens

    def _sentence_backfill(
        self, breached_embeddings, sentence_labels, shape, v_length, match_t=0.75, nontrivial_t=1e-2
    ):
        """Backfilling based only on sentence components. This is optional."""
        std, mean = torch.std_mean(breached_embeddings[:, :v_length], dim=-1, keepdim=True)
        normalized_components = (breached_embeddings[:, :v_length] - mean) / (std + 1e-10)
        seeds = torch.randn(shape[0], v_length)
        for sentence in range(shape[0]):
            seeds[sentence] = normalized_components[sentence_labels == sentence].median(dim=0).values
        unmixed_components = self._separate(normalized_components, seeds[sentence_labels])
        nontrivial_components = unmixed_components[unmixed_components.norm(dim=1) > nontrivial_t]
        component_ids = torch.arange(0, len(breached_embeddings))[unmixed_components.norm(dim=1) > nontrivial_t]
        log.info(f"Identified {(unmixed_components.norm(dim=1) < nontrivial_t).sum()} unique breaches.")

        _, counts = sentence_labels.unique(return_counts=True)
        free_positions = shape[1] - counts
        while free_positions.max() > 0:
            replicated_seeds = torch.repeat_interleave(seeds, free_positions, dim=0)
            replicated_labels = torch.repeat_interleave(torch.arange(0, shape[0]), free_positions, dim=0)
            order_breach_to_seed, selection_tensor, costs = self._match_embeddings(
                nontrivial_components, replicated_seeds, measure=self.cfg.matcher
            )
            # Accept assignments with higher correlation than 0.5
            matches = (costs > match_t).nonzero().squeeze(dim=-1)

            if len(matches) == 0:
                break
            else:
                log.info(f"Found {len(matches.nonzero())} new matches with avg. corr {costs[matches].mean()}")
            match_ids = component_ids[order_breach_to_seed][matches]
            breached_embeddings = torch.cat([breached_embeddings, breached_embeddings[match_ids]], dim=0)
            sentence_labels = torch.cat([sentence_labels, replicated_labels[selection_tensor][matches]], dim=0)

            # Decorrelate positions:
            # Compress slices into single-level to cope with Pytorch copying on sequential slices!
            ids = torch.arange(0, len(nontrivial_components))[order_breach_to_seed][matches]
            nontrivial_components[ids] = self._separate(
                nontrivial_components[ids], replicated_seeds[selection_tensor][matches]
            )
            nontrivial_breaches = nontrivial_components.norm(dim=1) > nontrivial_t
            nontrivial_components = nontrivial_components[nontrivial_breaches]
            component_ids = component_ids[nontrivial_breaches]

            _, counts = sentence_labels.unique(return_counts=True)
            free_positions = shape[1] - counts
        return breached_embeddings, sentence_labels

    def _separate(self, mixed_components, base_components):
        if self.cfg.separation == "subtraction":
            unmixed = mixed_components - base_components
        elif self.cfg.separation == "none":  # sanity check option
            unmixed = mixed_components.clone()
        elif self.cfg.separation == "decorrelation":
            dims = dict(dim=-1, keepdim=True)
            m_std, m_mean = torch.std_mean(mixed_components, **dims)
            b_std, b_mean = torch.std_mean(base_components, **dims)
            m_normed = (mixed_components - m_mean) / m_std
            b_normed = (base_components - b_mean) / b_std
            corr = (m_normed * b_normed).sum(**dims) / m_normed.norm(**dims) / b_normed.norm(**dims)
            unmixed_normed = m_normed - corr * b_normed
            unmixed = unmixed_normed * m_std + m_mean
        elif self.cfg.separation == "pca":  # Also decorrelation in a (not as nice) way
            N = mixed_components.shape[-1]
            A = torch.stack([mixed_components.view(-1, N), base_components.view(-1, N)], dim=1)
            _, _, V = torch.pca_lowrank(A - A.mean(dim=-1, keepdim=True), q=1, center=False, niter=20)
            unmixed = V[:, :, 0].view_as(mixed_components)
        else:
            raise ValueError(f"Invalid separation scheme {self.cfg.separation} given.")
        return unmixed

    def _supplement_from_full_vocabulary(self, recovered_tokens, costs, breached_without_positions, v_length, lookup):
        """Optionally: Match breached_without_positions to any embedding entries
        If the costs from the matching above are low, then this can recover lost tokens that were missed by
        .recover_token_information()
        """
        vocab_size = lookup["embedding"].weight.shape[0]
        all_token_ids = torch.arange(0, vocab_size, device=self.setup["device"])
        all_token_embeddings = lookup["norm_layer1"](lookup["embedding"](all_token_ids))
        all_token_embeddings = all_token_embeddings[:, v_length:-1]
        avg_costs = 0
        avg_new_corr = 0
        num_replaced_tokens = 0

        breached_tokens_np = breached_without_positions.detach().cpu().to(dtype=self.setup["dtype"]).numpy()
        all_tokens_embeddings_np = all_token_embeddings.detach()[1:].cpu().to(dtype=self.setup["dtype"]).numpy()
        for idx, entry in enumerate(breached_tokens_np):
            max_corr = self.vcorrcoef(all_tokens_embeddings_np, entry)
            if "abs" in self.cfg.matcher:
                val, loc = torch.as_tensor(max_corr).abs().max(dim=0)
            else:
                val, loc = torch.as_tensor(max_corr).max(dim=0)
            if val * self.cfg.embedding_token_weight > costs[idx]:
                num_replaced_tokens += 1
                avg_costs += costs[idx]
                avg_new_corr += val
                recovered_tokens[idx] = loc + 1
        if num_replaced_tokens > 0:
            log.info(
                f"Replaced {num_replaced_tokens} tokens with avg. corr {avg_costs / num_replaced_tokens} "
                f"with new tokens with avg corr {avg_new_corr / num_replaced_tokens}"
            )
        return recovered_tokens

    def _match_breaches_to_sentences(self, sentence_id_components, shape, algorithm="dynamic-threshold"):
        """Match (or rather cluster) the sentencen components into at maximum shape[0] sentences of length shape[1]"""
        if algorithm == "k-means":
            from k_means_constrained import KMeansConstrained

            clustering = KMeansConstrained(
                n_clusters=shape[0],
                size_min=0,
                size_max=min(shape[1], len(sentence_id_components)),
                init="k-means++",
                n_init=40,
                max_iter=900,
                tol=1e-6,
            )
            std, mean = torch.std_mean(sentence_id_components, dim=-1, keepdim=True)
            normalized_components = (sentence_id_components - mean) / (std + 1e-10)

            labels = clustering.fit_predict(normalized_components.double().numpy())
            sentence_labels = torch.as_tensor(labels, dtype=torch.long)

        elif algorithm == "k-medoids":
            from kmedoids import fasterpam

            corrs = torch.as_tensor(np.corrcoef(sentence_id_components.double().detach().numpy()))
            for trial in range(50):  # This is a hack to go around the missing constraint...
                medoids_result = fasterpam(corrs, shape[0])
                sentence_labels = torch.as_tensor(medoids_result.labels.astype(np.intc), dtype=torch.long)
                if sentence_labels.unique(return_counts=True)[1].max() <= shape[1]:
                    break
            assert sentence_labels.unique(return_counts=True)[1].max() <= shape[1], "Invalid Assignment in k-medoids"

        elif "dynamic-threshold" in algorithm:  # Allow for dynamic-threshold, dynamic-threshold-median and "normalized"
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
            label_ids = initial_labels[initial_labels != -1].unique()  # Skip over -1 (which is "unassigned")
            if "normalized" in algorithm:
                std, mean = torch.std_mean(sentence_id_components, dim=-1, keepdim=True)
                components = (sentence_id_components - mean) / (std + 1e-10)
            else:
                components = sentence_id_components
            for idx, group_label in enumerate(label_ids):
                if "median" in algorithm:
                    seeds[idx] = components[initial_labels == group_label].median(dim=0).values
                else:
                    seeds[idx] = components[initial_labels == group_label].mean(dim=0)

            # Replicate seeds to seq_length
            replicated_seeds = torch.repeat_interleave(seeds, shape[1], dim=0)

            # Recompute correlations based on these mean seeds
            order_breach_to_seed, _, _ = self._match_embeddings(replicated_seeds, components, measure=self.cfg.matcher)
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

        elif "pca" in algorithm:  # Allow for pca-direct and pca-assign (the default)
            A = sentence_id_components - sentence_id_components.mean(dim=-1, keepdim=True)
            # U, S, V = torch.pca_lowrank(A, q=shape[0], center=False, niter=20) # cannot handle q> min(m, n)
            U, S, V = torch.linalg.svd(A, full_matrices=False)
            log.info(f"Singular values in SVD: {S}")
            seeds = U[:, : shape[0]].T.matmul(A)  # all sign information is lost though
            if "direct" in algorithm:
                # this is the naive strategy, but it can break
                # and return more assignment per sentence than allowed
                sentence_labels = U[:, : shape[0]].abs().argmax(dim=-1)
            else:
                # Replicate seeds to seq_length
                replicated_seeds = torch.repeat_interleave(seeds, shape[1], dim=0)

                # Recompute correlations based on these mean seeds
                order_breach_to_seed, _, _ = self._match_embeddings(replicated_seeds, A, measure=self.cfg.matcher)
                sentence_labels = (order_breach_to_seed / shape[1]).to(dtype=torch.long)
            # Should use U later on to do a better collision detection

        else:
            raise ValueError(f"Invalid sentence algorithm {algorithm} given.")
        return sentence_labels

    def _match_embeddings(self, inputs, references, measure="corrcoef", fallbacks=None):
        if references.ndim == 1:
            references = references[None, :]
        if measure == "corrcoef":
            s, e = inputs.shape
            corr = np.corrcoef(inputs.detach().cpu().numpy(), references.detach().cpu().numpy())[s:, :s]
            corr[np.isnan(corr)] = 0
        elif measure == "abs-corrcoef":
            s, e = inputs.shape
            corr = np.abs(np.corrcoef(inputs.detach().cpu().numpy(), references.detach().cpu().numpy())[s:, :s])
            corr[np.isnan(corr)] = 0
        else:
            corr = references.matmul(inputs.T) / references.norm(dim=-1)[:, None] / inputs.norm(dim=-1)[None, :]
            corr = corr.detach().numpy()
        try:
            row_ind, col_ind = linear_sum_assignment(corr, maximize=True)
        except ValueError:
            log.info(f"ValueError from correlation matrix {corr}")
            if fallbacks is None:
                log.info("Returning trivial order...")
                row_ind, col_ind = list(range(corr.shape[0])), list(range(corr.shape[0]))
            else:
                log.info("Returning fallback order...")
                row_ind, col_ind = list(range(corr.shape[0])), fallbacks
        order_tensor = torch.as_tensor(col_ind, device=inputs.device, dtype=torch.long)
        selection_tensor = torch.as_tensor(row_ind, device=inputs.device, dtype=torch.long)
        costs = torch.as_tensor(corr[row_ind, col_ind], device=inputs.device, dtype=torch.float)
        return order_tensor, selection_tensor, costs

    def _compute_confidence_estimates(self, final_tokens, breached_embeddings, v_length, lookup):
        """Rough estimates how confident the attacker is that the token is correctly identified. This is uncalibrated confidence!.
        Or rather, a confidence of 1.0 is a good indicator that the token is correct, all lower confidences indicate mismatches."""
        vocab_size = lookup["embedding"].weight.shape[0]
        all_token_ids = torch.arange(0, vocab_size, device=self.setup["device"])
        all_token_embeddings = lookup["embedding"](all_token_ids)

        pure_positions = lookup["pos_encoder"](torch.zeros_like(lookup["embedding"](final_tokens))).view(
            -1, all_token_embeddings.shape[1]
        )
        estimated_word_embeddings = all_token_embeddings[final_tokens.view(-1)]
        estimated_final_embeddings = lookup["norm_layer1"](estimated_word_embeddings + pure_positions)[:, v_length:-1]
        # free_positions = estimated_final_embeddings.shape[0] - breached_embeddings.shape[0]
        # _, _, costs = self._match_embeddings(
        #     estimated_final_embeddings,
        #     breached_embeddings.repeat_interleave(free_positions // breached_embeddings.shape[0] + 2, dim=0),
        #     measure=self.cfg.matcher,
        # )
        costs = torch.zeros_like(final_tokens.view(-1), dtype=torch.float)
        for idx, embedding in enumerate(estimated_final_embeddings):
            if "abs" in self.cfg.matcher:
                costs[idx] = np.abs(self.vcorrcoef(breached_embeddings.numpy(), embedding.numpy())).max().item()
            else:
                costs[idx] = self.vcorrcoef(breached_embeddings.numpy(), embedding.numpy()).max().item()
        return costs.view_as(final_tokens)

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

        x_patched = self.closed_form_april(rec_models[0], shared_data[0], solver=self.cfg.solver).to(**self.setup)
        x = self.recover_patch(x_patched)
        inputs = torch.max(torch.min(x, (1 - self.dm) / self.ds), -self.dm / self.ds)

        # This attack always recovers just one sample!
        data = torch.zeros([len_data, *self.data_shape], **self.setup)
        data[0] = inputs
        reconstructed_data = dict(data=data, labels=labels)
        if "ClassAttack" in server_secrets:
            # The classattack secret knows which image was reconstructed
            true_num_data = server_secrets["ClassAttack"]["true_num_data"]
            reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
            reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = inputs
            reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
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
    def closed_form_april(model, shared_data, solver="gelsd"):
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
        z = torch.linalg.lstsq(A.T, b, driver=solver, rcond=None).solution
        z -= pos_embed
        z = z[1:]

        # recover img
        em_w = model.model.patch_embed.proj.weight.detach().double().cpu()
        em_w = em_w.reshape((em_w.shape[0], -1))
        em_b = model.model.patch_embed.proj.bias.detach().double().cpu()

        x = z - em_b
        log.info(f"Embedding Inversion:  ||A||={em_w.norm()}, ||b||={x.norm()}")
        x = torch.linalg.lstsq(em_w, x.T, driver=solver, rcond=None).solution
        x = x.reshape((3, -1, x.shape[-1]))
        x = x.transpose(1, 2)
        return x
