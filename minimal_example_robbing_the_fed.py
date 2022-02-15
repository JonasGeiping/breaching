"""
Example script to run an attack from this repository directly without simulation.

This is a quick example for the imprint module attack described in
- Fowl et al. "Robbing the Fed: Directly Obtaining Private Information in Federated Learning"

The jupyter notebooks have more details about the attack, but this code snippet is hopefully useful if you want
to check a model architecture and model gradients computed/defended in some shape or form without implementing
your model into the simulation.

All caveats apply. Make sure not to leak any unexpected information.
"""
import torch
import torchvision
from breaching.attacks.analytic_attack import ImprintAttacker
from breaching.cases.malicious_modifications.imprint import ImprintBlock
from collections import namedtuple


class data_cfg_default:
    modality = "vision"
    size = (1_281_167,)
    classes = 1000
    shape = (3, 224, 224)
    normalize = True
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


class attack_cfg_default:
    type = "analytic"
    attack_type = "imprint-readout"
    label_strategy = None  # Labels are not actually required for this attack
    normalize_gradients = False
    sort_by_bias = False
    text_strategy = "no-preprocessing"
    token_strategy = "decoder-bias"
    token_recovery = "from-limited-embedding"
    breach_reduction = "weight"
    impl = namedtuple("impl", ["dtype", "mixed_precision", "JIT"])("float", False, "")


transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=data_cfg_default.mean, std=data_cfg_default.std),
    ]
)


def main():
    setup = dict(device=torch.device("cpu"), dtype=torch.float)

    # This could be any model:
    model = torchvision.models.resnet18()
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    # It will be modified maliciously:
    block = ImprintBlock(data_cfg_default.shape, num_bins=4)
    model = torch.nn.Sequential(block, model)
    secret = dict(weight_idx=0, bias_idx=1, shape=tuple(data_cfg_default.shape), structure=block.structure)
    secrets = {"ImprintBlock": secret}

    # And your dataset:
    dataset = torchvision.datasets.ImageNet(root="~/data/imagenet", split="val", transform=transforms)
    datapoint, label = dataset[1200]  # This is the owl, just for the sake of this experiment
    labels = torch.as_tensor(label)[None, ...]

    # This is the attacker:
    attacker = ImprintAttacker(model, loss_fn, attack_cfg_default, setup)

    # ## Simulate an attacked FL protocol
    # Server-side computation:
    server_payload = [
        dict(
            parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()], metadata=data_cfg_default
        )
    ]
    # User-side computation:
    loss = loss_fn(model(datapoint[None, ...]), labels)
    shared_data = [
        dict(
            gradients=torch.autograd.grad(loss, model.parameters()),
            buffers=None,
            metadata=dict(num_data_points=1, labels=labels, local_hyperparams=None,),
        )
    ]

    # Attack:
    reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, secrets, dryrun=False)

    # Do some processing of your choice here. Maybe save the output image?


if __name__ == "__main__":
    main()
