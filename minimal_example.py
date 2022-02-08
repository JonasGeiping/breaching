"""
Example script to run attacks in this repository directly without simulation.
This can be useful if you want to check a model architecture and model gradients computed/defended in some shape or form
against some of the attacks implemented in this repository, without implementing your model into the simulation.

All caveats apply. Make sure not to leak any unexpected information.
"""
import torch
import torchvision
import breaching


class data_cfg_default:
    modality = "vision"
    size = (1_281_167,)
    classes = 1000
    shape = (3, 224, 224)
    normalize = True
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


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

    # This could be your model:
    model = torchvision.models.resnet152(pretrained=True)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    # And your dataset:
    dataset = torchvision.datasets.ImageNet(root="~/data/imagenet", split="val", transform=transforms)
    datapoint, label = dataset[1200]  # This is the owl, just for the sake of this experiment
    labels = torch.as_tensor(label)[None, ...]

    # This is the attacker:
    cfg_attack = breaching.get_attack_config("invertinggradients")
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg_attack, setup)

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
    reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, {}, dryrun=False)

    # Do some processing of your choice here. Maybe save the output image?


if __name__ == "__main__":
    main()
