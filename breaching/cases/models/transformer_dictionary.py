"""Dictionary lookup and identification of module names for different architectures."""
import torch


def lookup_module_names(model_name, model):
    """New architectures have to be registered here before the decepticons can know what is where."""

    lookup = dict()

    if "transformer" in model_name:  # These are the basic transformers from language_models.py
        assert model_name in ["transformer1", "transformer3", "transformer3f", "transformer3t", "transformerS"]

        lookup["loss"] = "causal"
        lookup["embedding"] = model.encoder
        lookup["pos_encoder"] = model.pos_encoder

        lookup["norm_layer0"] = torch.nn.Identity()  # This would be a norm before the MHA
        lookup["norm_layer1"] = model.transformer_encoder.layers[0].norm1

        lookup["first_attention"] = dict()
        lookup["first_attention"]["mode"] = "default"
        lookup["first_attention"]["in_proj_weight"] = model.transformer_encoder.layers[0].self_attn.in_proj_weight
        lookup["first_attention"]["in_proj_bias"] = model.transformer_encoder.layers[0].self_attn.in_proj_bias
        lookup["first_attention"]["out_proj_weight"] = model.transformer_encoder.layers[0].self_attn.out_proj.weight
        lookup["first_attention"]["out_proj_bias"] = model.transformer_encoder.layers[0].self_attn.out_proj.bias

        lookup["last_attention"] = dict()
        lookup["last_attention"]["mode"] = "default"
        lookup["last_attention"]["in_proj_weight"] = model.transformer_encoder.layers[-1].self_attn.in_proj_weight
        lookup["last_attention"]["in_proj_bias"] = model.transformer_encoder.layers[-1].self_attn.in_proj_bias
        lookup["last_attention"]["out_proj_weight"] = model.transformer_encoder.layers[-1].self_attn.out_proj.weight
        lookup["last_attention"]["out_proj_bias"] = model.transformer_encoder.layers[-1].self_attn.out_proj.bias

        first_linear_layers, second_linear_layers, unused_mhas = [], [], []  # collecting all the imprint layers
        for i, layer in enumerate(model.transformer_encoder.layers):
            first_linear_layers.append(layer.linear1)
            second_linear_layers.append(layer.linear2)
            if i > 0 and i < len(model.transformer_encoder.layers) - 1:  # all intermediate attention blocks
                unused_mhas.append(layer.self_attn.out_proj)

        lookup["first_linear_layers"] = first_linear_layers
        lookup["second_linear_layers"] = second_linear_layers
        lookup["unused_mha_outs"] = unused_mhas

        hidden_dim, embedding_dim = first_linear_layers[0].weight.shape
        ff_transposed = False
        lookup["dimensions"] = hidden_dim, embedding_dim, ff_transposed
    elif "gpt2" in model_name:  # This is huggingface's gpt2 implementation
        assert model_name in ["gpt2", "gpt2S"]

        lookup["loss"] = "causal"
        lookup["embedding"] = model.model.transformer.wte
        lookup["pos_encoder"] = PositionalContainer(model.model.transformer.wpe)

        lookup["norm_layer0"] = torch.nn.Identity()  # Better disabled? model.model.transformer.h[0].ln_1
        lookup["norm_layer1"] = model.model.transformer.h[0].ln_2

        lookup["first_attention"] = dict()
        lookup["first_attention"]["mode"] = "default"
        lookup["first_attention"]["in_proj_weight"] = model.model.transformer.h[0].attn.c_attn.weight
        lookup["first_attention"]["in_proj_bias"] = model.model.transformer.h[0].attn.c_attn.bias
        lookup["first_attention"]["out_proj_weight"] = model.model.transformer.h[0].attn.c_proj.weight
        lookup["first_attention"]["out_proj_bias"] = model.model.transformer.h[0].attn.c_proj.bias

        lookup["last_attention"] = dict()
        lookup["last_attention"]["mode"] = "default"
        lookup["last_attention"]["in_proj_weight"] = model.model.transformer.h[-1].attn.c_attn.weight
        lookup["last_attention"]["in_proj_bias"] = model.model.transformer.h[-1].attn.c_attn.bias
        lookup["last_attention"]["out_proj_weight"] = model.model.transformer.h[-1].attn.c_proj.weight
        lookup["last_attention"]["out_proj_bias"] = model.model.transformer.h[-1].attn.c_proj.bias

        first_linear_layers, second_linear_layers, unused_mhas = [], [], []  # collecting all the imprint layers
        for i, layer in enumerate(model.model.transformer.h):
            first_linear_layers.append(layer.mlp.c_fc)
            second_linear_layers.append(layer.mlp.c_proj)
            if i > 0 and i < len(model.model.transformer.h) - 1:
                unused_mhas.append(layer.attn.c_proj)

        lookup["first_linear_layers"] = first_linear_layers
        lookup["second_linear_layers"] = second_linear_layers
        lookup["unused_mha_outs"] = unused_mhas

        hidden_dim, embedding_dim = first_linear_layers[0].weight.T.shape
        ff_transposed = True
        lookup["dimensions"] = hidden_dim, embedding_dim, ff_transposed

    elif "bert" in model_name or "BERT" in model_name:
        assert model_name in [
            "bert-base-uncased",
            "bert-large-uncased",
            "bert-small-uncased",
            "bert-sanity-check",
            "huawei-noah/TinyBERT_General_4L_312D",
        ]
        bert = model.model.bert
        lookup["loss"] = "mlm"
        lookup["embedding"] = bert.embeddings.word_embeddings
        lookup["pos_encoder"] = PositionalContainer(bert.embeddings.position_embeddings)

        lookup["norm_layer0"] = bert.embeddings.LayerNorm  # This would be a norm before the MHA
        lookup["norm_layer1"] = bert.encoder.layer[0].output.LayerNorm

        lookup["first_attention"] = dict()
        lookup["first_attention"]["mode"] = "bert"
        lookup["first_attention"]["query"] = bert.encoder.layer[0].attention.self.query
        lookup["first_attention"]["key"] = bert.encoder.layer[0].attention.self.key
        lookup["first_attention"]["value"] = bert.encoder.layer[0].attention.self.value
        lookup["first_attention"]["output"] = bert.encoder.layer[0].attention.output.dense

        lookup["last_attention"] = dict()
        lookup["last_attention"]["mode"] = "bert"
        lookup["last_attention"]["query"] = bert.encoder.layer[-1].attention.self.query
        lookup["last_attention"]["key"] = bert.encoder.layer[-1].attention.self.key
        lookup["last_attention"]["value"] = bert.encoder.layer[-1].attention.self.value
        lookup["last_attention"]["output"] = bert.encoder.layer[-1].attention.output.dense

        first_linear_layers, second_linear_layers, unused_mhas = [], [], []  # collecting all the imprint layers
        for i, layer in enumerate(bert.encoder.layer):
            first_linear_layers.append(layer.intermediate.dense)
            second_linear_layers.append(layer.output.dense)
            if i > 0 and i < len(bert.encoder.layer) - 1:
                unused_mhas.append(layer.attention.output.dense)

        lookup["first_linear_layers"] = first_linear_layers
        lookup["second_linear_layers"] = second_linear_layers
        lookup["unused_mha_outs"] = unused_mhas

        hidden_dim, embedding_dim = first_linear_layers[0].weight.shape
        ff_transposed = False
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
        embedding_parameter_idx = -2 if model_name == "transformer3t" else -3
        decoder_bias_parameter_idx = -1
    elif "gpt2" in model_name:
        embedding_parameter_idx = 0
        decoder_bias_parameter_idx = None  # No decoder bias!
    elif "bert" in model_name or "BERT" in model_name:
        embedding_parameter_idx = 0
        decoder_bias_parameter_idx = -5  # No decoder bias!
    else:
        raise ValueError(f"Unknown architecture {model_name} not registered in index lookup table!")

    return embedding_parameter_idx, decoder_bias_parameter_idx
