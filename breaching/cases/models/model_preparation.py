"""Helper code to instantiate various models."""

import torchvision


def construct_model(cfg_model, cfg_data, pretrained=False):
    """Construct the neural net that is used."""
    if cfg_model == 'ResNet152':
        model = torchvision.models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f'Invalid model spec {cfg_model} given.')

    return model
