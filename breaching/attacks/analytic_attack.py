"""Simple analytic attack that works for (dumb) fully connected models."""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment  # Better than greedy search
from collections import defaultdict

from .base_attack import _BaseAttacker


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

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """This is somewhat hard-coded for images, but that is not a necessity."""
        # Initialize stats module for later usage:
        rec_models, tokens, stats = self.prepare_attack(server_payload, shared_data)
        len_data = shared_data[0]["metadata"]["num_data_points"]  # Could be guessed as well

        if "ImprintBlock" in server_secrets.keys():
            weight_idx = server_secrets["ImprintBlock"]["weight_idx"]
            bias_idx = server_secrets["ImprintBlock"]["bias_idx"]
            data_shape = server_secrets["ImprintBlock"]["data_shape"]
        else:
            raise ValueError(f"No imprint hidden in model {rec_models[0]} according to server.")

        leaked_tokens = self.recover_token_information(shared_data, server_payload, rec_models).view(-1)

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
        if with_pos.shape[0] < len_data:
            # Some positional tokens are missing, we substitute with random vectors:
            random_embeds = torch.zeros(len_data - with_pos.shape[0], with_pos.shape[1])
            with_pos = torch.cat([with_pos, random_embeds])

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
        PAD_token = 220
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
            max_corr = 0
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
