"""Implement functions for the interface.

* cases.construct_case
"""
import torch


from .data import construct_dataloader
from .models import construct_model

from .users import UserSingleStep
from .servers import HonestServer, MaliciousModelServer, MaliciousParameterServer, PathParameterServer, StackParameterServer


def construct_case(cfg_case, setup, dryrun=False):
    """Construct training splits (from which to draw examples) and model states and place into user and server objects."""

    # Load multiple splits only if necessary
    # So that I don't need to have the ImageNet training set on my laptop:
    dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, cfg_case.examples_from_split, dryrun=dryrun)
    model = construct_model(cfg_case.model, cfg_case.data, pretrained='trained' in cfg_case.server.model_state)
    loss = torch.nn.CrossEntropyLoss()
    if cfg_case.server.has_external_data:
        external_dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, 'training', dryrun=dryrun)
    else:
        external_dataloader = None

    if cfg_case.server.name == 'honest_but_curious':
        server = HonestServer(model, loss, cfg_case, setup, external_dataloader=external_dataloader)
    elif cfg_case.server.name == 'malicious_model':
        server = MaliciousModelServer(model, loss, cfg_case, setup, external_dataloader=external_dataloader)
    elif cfg_case.server.name == 'malicious_parameters':
        server = MaliciousParameterServer(model, loss, cfg_case, setup, external_dataloader=external_dataloader)
    elif cfg_case.server.name == 'path_parameters':
        server = PathParameterServer(model, loss, cfg_case, setup, external_dataloader=external_dataloader)
    elif cfg_case.server.name == 'stack_parameters':
        server = StackParameterServer(model, loss, cfg_case, setup, external_dataloader=external_dataloader)
    else:
        raise ValueError(f'Invalid server settings {cfg_case.server} given.')

    model = server.prepare_model()
    num_params, num_buffers = sum([p.numel() for p in model.parameters()]), sum([b.numel() for b in model.buffers()])
    target_information = cfg_case.user.num_data_points * torch.as_tensor(cfg_case.data.shape).prod()
    print(f'Model architecture {model.__class__} loaded with {num_params:,} parameters and {num_buffers:,} buffers.')
    print(f'Overall this is a data ratio of {cfg_case.num_queries * num_params / target_information:7.0f}:1 '
          f'for target shape {[cfg_case.user.num_data_points, *cfg_case.data.shape]} given that num_queries={cfg_case.num_queries}.')

    if cfg_case.user.num_local_updates == 1:
        # The user will deepcopy this model to have their own
        user = UserSingleStep(model, loss, dataloader, setup, **cfg_case.user)
    else:
        raise ValueError('User specifications not implemented.')

    return user, server
