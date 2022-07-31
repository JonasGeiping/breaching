"""Helper code to instantiate various models."""

import torch
import torchvision

from collections import OrderedDict

from .resnets import ResNet, resnet_depths_to_config
from .densenets import DenseNet, densenet_depths_to_config
from .nfnets import NFNet
from .vgg import VGG

from .language_models import RNNModel, TransformerModel, LinearModel
from .losses import CausalLoss, MLMLoss, MostlyCausalLoss


def construct_model(cfg_model, cfg_data, pretrained=True, **kwargs):
    if cfg_data.modality == "vision":
        model = _construct_vision_model(cfg_model, cfg_data, pretrained, **kwargs)
    elif cfg_data.modality == "text":
        model = _construct_text_model(cfg_model, cfg_data, pretrained, **kwargs)
    else:
        raise ValueError(f"Invalid data modality {cfg_data.modality}")
    # Save nametag for printouts later:
    model.name = cfg_model

    # Choose loss function according to data and model:
    if "classification" in cfg_data.task:
        loss_fn = torch.nn.CrossEntropyLoss()
    elif "causal-lm-sanity" in cfg_data.task:
        loss_fn = MostlyCausalLoss()
    elif "causal-lm" in cfg_data.task:
        loss_fn = CausalLoss()
    elif "masked-lm" in cfg_data.task:
        loss_fn = MLMLoss(vocab_size=cfg_data.vocab_size)
    else:
        raise ValueError(f"No loss function registered for task {cfg_data.task}.")
    loss_fn = torch.jit.script(loss_fn)
    return model, loss_fn


def _construct_text_model(cfg_model, cfg_data, pretrained=True, **kwargs):
    if cfg_model == "transformer3f":
        # This is the transformer from "A field guide to federated learning"
        """
        we train a modified 3-layer Transformer model [250],
        where the dimension of the token embeddings is 96, and the hidden dimension of the feed-forward
        network (FFN) block is 1536. We use 8 heads for the multi-head attention, where each head is based
        on 12-dimensional (query, key, value) vectors. We use ReLU activation and set dropout rate to 0.1.
        """
        # For simplicity the dropout is disabled for now
        # the 12-dim query is 96/8 = 12
        model = TransformerModel(
            ntokens=cfg_data.vocab_size, ninp=96, nhead=8, nhid=1536, nlayers=3, dropout=0, positional_embedding="fixed"
        )
    elif cfg_model == "transformer3":
        # Same as above but with learnable positional embeddings
        model = TransformerModel(
            ntokens=cfg_data.vocab_size,
            ninp=96,
            nhead=8,
            nhid=1536,
            nlayers=3,
            dropout=0,
            positional_embedding="learnable",
        )
    elif cfg_model == "transformer3t":
        # Same as above but with learnable positional embeddings and tied weights
        model = TransformerModel(
            ntokens=cfg_data.vocab_size,
            ninp=96,
            nhead=8,
            nhid=1536,
            nlayers=3,
            dropout=0,
            positional_embedding="learnable",
            tie_weights=True,
        )
    elif cfg_model == "transformer1":
        # This is our initial sanity check transformer:
        model = TransformerModel(ntokens=cfg_data.vocab_size, ninp=200, nhead=1, nhid=200, nlayers=1, dropout=0)
    elif cfg_model == "transformerS":
        # A wide sanity-check transformer
        model = TransformerModel(ntokens=cfg_data.vocab_size, ninp=512, nhead=1, nhid=512, nlayers=1, dropout=0)
    elif cfg_model == "LSTM":
        # This is the LSTM from "LEARNING DIFFERENTIALLY PRIVATE RECURRENT LANGUAGE MODELS"
        """
        word s t is mapped to an embedding vector e t ∈ R 96
        by looking up the word in the model’s vocabulary. The e t is composed with the state emitted by
        the model in the previous time step s t−1 ∈ R 256 to emit a new state vector s t and an “output
        embedding” o t ∈ R 96 .
        """
        model = RNNModel("LSTM", cfg_data.vocab_size, ninp=96, nhid=96, nlayers=1, dropout=0.0, tie_weights=True)
    elif cfg_model == "linear":
        model = LinearModel(cfg_data.vocab_size, embedding_size=200)
    else:
        try:
            from transformers import (
                AutoModelForMaskedLM,
                AutoModelForPreTraining,
                AutoModelForSequenceClassification,
                AutoConfig,
            )

            if cfg_data.task == "masked-lm":
                auto_class = AutoModelForMaskedLM
            elif cfg_data.task == "classification":
                auto_class = AutoModelForSequenceClassification
            else:
                auto_class = AutoModelForPreTraining
            # Make sure to use the matching tokenizer and vocab_size!
            if cfg_model == "gpt2S":
                cfg_model = "gpt2"
                extra_args = dict(activation_function="relu", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
            elif cfg_model == "bert-sanity-check":
                cfg_model = "bert-base-uncased"
                extra_args = dict(hidden_act="relu", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
            else:
                extra_args = dict()
            if pretrained:
                model = auto_class.from_pretrained(cfg_model, **extra_args)
            else:
                hf_cfg = AutoConfig.from_pretrained(cfg_model, **extra_args)
                model = auto_class.from_config(hf_cfg)
            # model.transformer.h[0].attn.scale_attn_weights = False
            if model.config.vocab_size != cfg_data.vocab_size:
                model.resize_token_embeddings(new_num_tokens=cfg_data.vocab_size)
            model = HuggingFaceContainer(model)
        except OSError as error_msg:
            raise ValueError(f"Invalid huggingface model {cfg_model} given: {error_msg}")
    return model


class HuggingFaceContainer(torch.nn.Module):
    """Wrap huggingface models for a unified interface. Ugh."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        if "inputs" in kwargs:
            kwargs["input_ids"] = kwargs.pop("inputs")
        if "input_ids" not in kwargs:
            kwargs["input_ids"] = args[0]
        if kwargs["input_ids"].dtype != torch.long:
            kwargs["inputs_embeds"] = kwargs.pop("input_ids")
        outputs = self.model(**kwargs)
        return outputs["logits"] if "logits" in outputs else outputs["prediction_logits"]


class VisionContainer(torch.nn.Module):
    """We'll use a container to catch extra attributes and allow for usage with model(**data)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, **kwargs):
        return self.model(inputs)


def _construct_vision_model(cfg_model, cfg_data, pretrained=True, **kwargs):
    """Construct the neural net that is used."""
    channels = cfg_data.shape[0]
    classes = cfg_data.classes

    if "ImageNet" in cfg_data.name:
        try:
            model = getattr(torchvision.models, cfg_model.lower())(pretrained=pretrained)
            try:
                # Try to adjust the linear layer and fill with previous data
                fc = torch.nn.Linear(model.fc.in_features, classes)
                if pretrained:
                    fc.weight.data = model.fc.weight[:classes]
                    fc.bias.data = model.fc.bias[:classes]
                model.fc = fc
            except AttributeError:
                pass
        except AttributeError:
            if "nfnet" in cfg_model:
                model = NFNet(
                    channels,
                    classes,
                    variant="F0",
                    stochdepth_rate=0.25,
                    alpha=0.2,
                    se_ratio=0.5,
                    activation="ReLU",
                    stem="ImageNet",
                    use_dropout=True,
                )
            elif "resnet101wsl" in cfg_model:
                model = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
            elif "resnet50swsl" in cfg_model:
                model = torch.hub.load("facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_swsl")
            elif "resnet50ssl" in cfg_model:
                model = torch.hub.load("facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_ssl")
            elif "resnetmoco" in cfg_model:
                model = torchvision.models.resnet50(pretrained=False)
                if pretrained:
                    # url = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar'
                    # url = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar'
                    url = "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.pth.tar"
                    state_dict = torch.hub.load_state_dict_from_url(
                        url, progress=True, map_location=torch.device("cpu")
                    )["state_dict"]
                    for key in list(state_dict.keys()):
                        val = state_dict.pop(key)
                        # sanitized_key = key.replace('module.encoder_q.', '') # for mocov2
                        sanitized_key = key.replace("module.", "")
                        state_dict[sanitized_key] = val

                    model.load_state_dict(state_dict, strict=True)  # The fc layer is not actually loaded here
            elif "vit_base_april" in cfg_model:
                import timm  # lazily import

                # timm models are listed at https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
                model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
                model.blocks[0] = ModifiedBlock(model.blocks[0])
            elif "vit_small_april" in cfg_model:
                import timm

                model = timm.create_model("vit_small_patch16_224", pretrained=pretrained)
                model.blocks[0] = ModifiedBlock(model.blocks[0])
            elif "vit_base" in cfg_model:
                import timm  # lazily import

                # timm models are listed at https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
                model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
            elif "vit_small" in cfg_model:
                import timm

                model = timm.create_model("vit_small_patch16_224", pretrained=pretrained)

            elif "linear" == cfg_model:
                input_dim = cfg_data.shape[0] * cfg_data.shape[1] * cfg_data.shape[2]
                model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_dim, classes))
            elif "none" == cfg_model:
                model = torch.nn.Sequential(torch.nn.Flatten(), _Select(classes))
            else:
                raise ValueError(f"Could not find ImageNet model {cfg_model} in torchvision.models or custom models.")
    else:
        # CIFAR Model from here:
        if "resnetgn" in cfg_model.lower():
            block, layers = resnet_depths_to_config(int("".join(filter(str.isdigit, cfg_model))))
            model = ResNet(
                block,
                layers,
                channels,
                classes,
                stem="CIFAR",
                convolution_type="Standard",
                nonlin="ReLU",
                norm="groupnorm4th",
                downsample="B",
                width_per_group=16 if len(layers) < 4 else 64,
                zero_init_residual=False,
            )
        elif "resnet" in cfg_model.lower():
            if "-" in cfg_model.lower():  # Hacky way to separate ResNets from wide ResNets which are e.g. 28-10
                depth = int("".join(filter(str.isdigit, cfg_model.split("-")[0])))
                width = int("".join(filter(str.isdigit, cfg_model.split("-")[1])))
            else:
                depth = int("".join(filter(str.isdigit, cfg_model)))
                width = 1
            block, layers = resnet_depths_to_config(depth)
            model = ResNet(
                block,
                layers,
                channels,
                classes,
                stem="CIFAR",
                convolution_type="Standard",
                nonlin="ReLU",
                norm="BatchNorm2d",
                downsample="B",
                width_per_group=(16 if len(layers) < 4 else 64) * width,
                zero_init_residual=False,
            )
        elif "densenet" in cfg_model.lower():
            growth_rate, block_config, num_init_features = densenet_depths_to_config(
                int("".join(filter(str.isdigit, cfg_model)))
            )
            model = DenseNet(
                growth_rate=growth_rate,
                block_config=block_config,
                num_init_features=num_init_features,
                bn_size=4,
                drop_rate=0,
                channels=channels,
                num_classes=classes,
                memory_efficient=False,
                norm="BatchNorm2d",
                nonlin="ReLU",
                stem="CIFAR",
                convolution_type="Standard",
            )
        elif "vgg" in cfg_model.lower():
            model = VGG(
                cfg_model,
                in_channels=channels,
                num_classes=classes,
                norm="BatchNorm2d",
                nonlin="ReLU",
                head="CIFAR",
                convolution_type="Standard",
                drop_rate=0,
                classical_weight_init=True,
            )
        elif "linear" in cfg_model:
            input_dim = cfg_data.shape[0] * cfg_data.shape[1] * cfg_data.shape[2]
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_dim, classes))
        elif "nfnet" in cfg_model:
            model = NFNet(
                channels,
                classes,
                variant="F0",
                stochdepth_rate=0.25,
                alpha=0.2,
                se_ratio=0.5,
                activation="ReLU",
                stem="CIFAR",
                use_dropout=True,
            )
        elif "convnet-trivial" == cfg_model.lower():
            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("conv", torch.nn.Conv2d(channels, 3072, 3, stride=1, padding=1)),
                        ("relu", torch.nn.ReLU(inplace=True)),
                        ("pool", torch.nn.AdaptiveAvgPool2d(1)),
                        ("flatten", torch.nn.Flatten()),
                        ("linear", torch.nn.Linear(3072, classes)),
                    ]
                )
            )
        elif "convnetsmall" == cfg_model.lower():
            model = ConvNetSmall(width=256, num_channels=channels, num_classes=classes)
        elif "convnet" == cfg_model.lower():
            model = ConvNet(width=64, num_channels=channels, num_classes=classes)
        elif "convnet_beyond" == cfg_model.lower():
            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", torch.nn.Conv2d(channels, 32, 3, stride=2, padding=1)),
                        ("relu0", torch.nn.LeakyReLU()),
                        ("conv2", torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)),
                        ("relu1", torch.nn.LeakyReLU()),
                        ("conv3", torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)),
                        ("relu2", torch.nn.LeakyReLU()),
                        ("conv4", torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)),
                        ("relu3", torch.nn.LeakyReLU()),
                        ("flatt", torch.nn.Flatten()),
                        ("linear0", torch.nn.Linear(12544, 12544)),
                        ("relu4", torch.nn.LeakyReLU()),
                        ("linear1", torch.nn.Linear(12544, classes)),
                        ("softmax", torch.nn.Softmax(dim=1)),
                    ]
                )
            )
        elif "lenet_zhu" == cfg_model.lower():
            model = LeNetZhu(num_channels=channels, num_classes=classes)
        elif "cnn6" == cfg_model.lower():
            # This is the model from R-GAP:
            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("layer0", torch.nn.Conv2d(channels, 12, kernel_size=4, padding=2, stride=2, bias=False)),
                        ("act0", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer1", torch.nn.Conv2d(12, 36, kernel_size=3, padding=1, stride=2, bias=False)),
                        ("act1", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer2", torch.nn.Conv2d(36, 36, kernel_size=3, padding=1, stride=1, bias=False)),
                        ("act2", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer3", torch.nn.Conv2d(36, 36, kernel_size=3, padding=1, stride=1, bias=False)),
                        ("act3", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer4", torch.nn.Conv2d(36, 64, kernel_size=3, padding=1, stride=2, bias=False)),
                        ("act4", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer5", torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=False)),
                        ("flatten", torch.nn.Flatten()),
                        ("act5", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("fc", torch.nn.Linear(3200, classes, bias=True)),
                    ]
                )
            )
        elif cfg_model == "MLP":
            width = 1024
            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("flatten", torch.nn.Flatten()),
                        ("linear0", torch.nn.Linear(3072, width)),
                        ("relu0", torch.nn.ReLU()),
                        ("linear1", torch.nn.Linear(width, width)),
                        ("relu1", torch.nn.ReLU()),
                        ("linear2", torch.nn.Linear(width, width)),
                        ("relu2", torch.nn.ReLU()),
                        ("linear3", torch.nn.Linear(width, classes)),
                    ]
                )
            )
        else:
            raise ValueError("Model could not be found.")

    return VisionContainer(model)


class ConvNetSmall(torch.nn.Module):
    """ConvNet without BN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv0", torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
                    ("relu0", torch.nn.ReLU()),
                    ("conv1", torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
                    ("relu1", torch.nn.ReLU()),
                    ("conv2", torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, stride=2, padding=1)),
                    ("relu2", torch.nn.ReLU()),
                    ("pool0", torch.nn.MaxPool2d(3)),
                    ("conv3", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, stride=2, padding=1)),
                    ("relu3", torch.nn.ReLU()),
                    ("pool1", torch.nn.AdaptiveAvgPool2d(1)),
                    ("flatten", torch.nn.Flatten()),
                    ("linear", torch.nn.Linear(4 * width, num_classes)),
                ]
            )
        )

    def forward(self, input):
        return self.model(input)


class ConvNet(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv0", torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
                    ("bn0", torch.nn.BatchNorm2d(1 * width)),
                    ("relu0", torch.nn.ReLU()),
                    ("conv1", torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
                    ("bn1", torch.nn.BatchNorm2d(2 * width)),
                    ("relu1", torch.nn.ReLU()),
                    ("conv2", torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
                    ("bn2", torch.nn.BatchNorm2d(2 * width)),
                    ("relu2", torch.nn.ReLU()),
                    ("conv3", torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn3", torch.nn.BatchNorm2d(4 * width)),
                    ("relu3", torch.nn.ReLU()),
                    ("conv4", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn4", torch.nn.BatchNorm2d(4 * width)),
                    ("relu4", torch.nn.ReLU()),
                    ("conv5", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn5", torch.nn.BatchNorm2d(4 * width)),
                    ("relu5", torch.nn.ReLU()),
                    ("pool0", torch.nn.MaxPool2d(3)),
                    ("conv6", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn6", torch.nn.BatchNorm2d(4 * width)),
                    ("relu6", torch.nn.ReLU()),
                    ("conv7", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn7", torch.nn.BatchNorm2d(4 * width)),
                    ("relu7", torch.nn.ReLU()),
                    ("pool1", torch.nn.MaxPool2d(3)),
                    ("flatten", torch.nn.Flatten()),
                    ("linear", torch.nn.Linear(36 * width, num_classes)),
                ]
            )
        )

    def forward(self, input):
        return self.model(input)


class LeNetZhu(torch.nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""

    def __init__(self, num_classes=10, num_channels=3):
        """3-Layer sigmoid Conv with large linear layer."""
        super().__init__()
        act = torch.nn.Sigmoid
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            torch.nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            torch.nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = torch.nn.Sequential(torch.nn.Linear(768, num_classes))
        for module in self.modules():
            self.weights_init(module)

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


class _Select(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, : self.n]


class ModifiedBlock(torch.nn.Module):
    def __init__(self, old_Block):
        super().__init__()
        self.attn = old_Block.attn
        self.drop_path = old_Block.drop_path
        self.norm2 = old_Block.norm2
        self.mlp = old_Block.mlp

    def forward(self, x):
        x = self.attn(x)
        x = self.drop_path(self.mlp((self.norm2(x))))
        return x
