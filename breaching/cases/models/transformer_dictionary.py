"""Dictionary lookup and identification of module names for different architectures."""
import torch


def lookup_module_names(model_name, model):
    """New architectures have to be registered here before the decepticons can know what is where."""

    lookup = dict()
    if "transformer" in model_name:  # These are the basic transformers from language_models.py
        assert model_name in ["transformer1", "transformer3", "transformer3f", "transformer3t", "transformerS"]

        lookup["embedding"] = model.encoder
        lookup["pos_encoder"] = model.pos_encoder

        lookup["norm_layer0"] = torch.nn.Identity()  # This would be a norm before the MHA
        lookup["norm_layer1"] = model.transformer_encoder.layers[0].norm1

        lookup["attention_layer"] = dict()
        lookup["attention_layer"]["in_proj_weight"] = model.transformer_encoder.layers[0].self_attn.in_proj_weight
        lookup["attention_layer"]["in_proj_bias"] = model.transformer_encoder.layers[0].self_attn.in_proj_bias
        lookup["attention_layer"]["out_proj_weight"] = model.transformer_encoder.layers[0].self_attn.out_proj.weight
        lookup["attention_layer"]["out_proj_bias"] = model.transformer_encoder.layers[0].self_attn.out_proj.bias

        first_linear_layers, second_linear_layers, unused_mhas = [], [], []  # collecting all the imprint layers
        for i, layer in enumerate(model.transformer_encoder.layers):
            first_linear_layers.append(layer.linear1)
            second_linear_layers.append(layer.linear2)
            if i != 0:
                unused_mhas.append(layer.self_attn.out_proj)

        lookup["first_linear_layers"] = first_linear_layers
        lookup["second_linear_layers"] = second_linear_layers
        lookup["unused_mha_outs"] = unused_mhas

        hidden_dim, embedding_dim = first_linear_layers[0].weight.shape
        ff_transposed = False
        lookup["dimensions"] = hidden_dim, embedding_dim, ff_transposed
    elif "gpt2" in model_name:  # This is huggingface's gpt2 implementation
        assert model_name in ["gpt2"]

        lookup["embedding"] = model.model.transformer.wte
        lookup["pos_encoder"] = PositionalContainer(model.model.transformer.wpe)

        lookup["norm_layer0"] = model.model.transformer.h[0].ln_1
        lookup["norm_layer1"] = model.model.transformer.h[0].ln_2

        lookup["attention_layer"] = dict()
        lookup["attention_layer"]["in_proj_weight"] = model.model.transformer.h[0].attn.c_attn.weight
        lookup["attention_layer"]["in_proj_bias"] = model.model.transformer.h[0].attn.c_attn.bias
        lookup["attention_layer"]["out_proj_weight"] = model.model.transformer.h[0].attn.c_proj.weight
        lookup["attention_layer"]["out_proj_bias"] = model.model.transformer.h[0].attn.c_proj.bias

        first_linear_layers, second_linear_layers, unused_mhas = [], [], []  # collecting all the imprint layers
        for i, layer in enumerate(model.model.transformer.h):
            first_linear_layers.append(layer.mlp.c_fc)
            second_linear_layers.append(layer.mlp.c_proj)
            if i != 0:
                unused_mhas.append(layer.attn.c_proj)

        lookup["first_linear_layers"] = first_linear_layers
        lookup["second_linear_layers"] = second_linear_layers
        lookup["unused_mha_outs"] = unused_mhas

        hidden_dim, embedding_dim = first_linear_layers[0].weight.T.shape
        ff_transposed = True
        lookup["dimensions"] = hidden_dim, embedding_dim, ff_transposed
    else:
        raise ValueError(f"Unknown architecture {model_name} not registered in module lookup table!")

    return lookup


class PositionalContainer(torch.nn.Module):
    """Container for a learnable positional embedding."""

    def __init__(self, pos_encoder):
        super().__init__()
        self.embedding = pos_encoder

    def forward(self, input_embeddings):
        """This is a batch-first implementation"""
        position_ids = torch.arange(input_embeddings.shape[1], device=self.embedding.weight.device)
        position_embeddings = self.embedding(position_ids[None, :])
        return input_embeddings + position_embeddings


def lookup_grad_indices(model_name):
    """Which index in the list of grads corresponds to embedding weight and which to last linear layer bias?"""
    if "transformer" in model_name:  # This lookup is not automated :> Add new models here
        embedding_parameter_idx = -2 if rec_models[0].name == "transformer3t" else -3
        decoder_bias_parameter_idx = -1
    elif "gpt2" in model_name:
        embedding_parameter_idx = 0
        decoder_bias_parameter_idx = None  # No decoder bias!
    else:
        raise ValueError(f"Unknown architecture {model_name} not registered in module lookup table!")
