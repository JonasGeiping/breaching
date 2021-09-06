"""Implement functions for the interface.

* cases.construct_case
"""
import torch


from .data import construct_dataloader
from .models import construct_model

from .users import UserSingleStep
from .servers import HonestServer

def construct_case(cfg_case, setup, dryrun=False):
    """Construct training splits (from which to draw examples) and model states and place into user and server objects."""

    # Load multiple splits only if necessary
    # So that I don't need to have the ImageNet training set on my laptop:
    dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, cfg_case.examples_from_split, dryrun=dryrun)
    model = construct_model(cfg_case.model, cfg_case.data, pretrained=cfg_case.server.model_state == 'trained')
    loss = torch.nn.CrossEntropyLoss()

    if cfg_case.server.name == 'honest_but_curious':
        server = HonestServer(model, loss, cfg_case.server.model_state, cfg_case.num_queries, cfg_case.data)
    else:
        raise ValueError(f'Invalid server settings {cfg_case.server} given.')

    if cfg_case.user.num_local_updates == 1:
        user = UserSingleStep(model, loss, dataloader, setup, **cfg_case.user)  # The user will deepcopy this model to have their own
    else:
        raise ValueError('User specifications not implemented.')

    return user, server
