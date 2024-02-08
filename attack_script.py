import torch
import breaching
from torchvision import models

def setup_attack(cfg, torch_model=None):
    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
    print(setup)

    #setup all customisable parameters

    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    # breaching.utils.overview(server, user, attacker)


    if torch_model is not None:
        model = torch_model
    
    if not check_image_size(model, cfg.case.data.shape):
        raise ValueError("Mismatched dimensions")
    
    return setup, user, server, attacker, model, loss_fn

def perform_attack(cfg, setup, user, server, attacker, model, loss_fn):
    server_payload = server.distribute_payload()
    shared_data, true_user_data = user.compute_local_updates(server_payload)
    breaching.utils.overview(server, user, attacker)

    user.plot(true_user_data, saveFile="true_data")
    print("reconstructing attack")
    reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)
    user.plot(reconstructed_user_data, saveFile="reconstructed_data")
    return reconstructed_user_data, true_user_data, server_payload
    
def get_metrics(reconstructed_user_data, true_user_data, server_payload, server, cfg, setup):
    metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload], 
                                     server.model, order_batch=True, compute_full_iip=False, 
                                     cfg_case=cfg.case, setup=setup, compute_lpips=True)
    return metrics
    
def check_image_size(model, shape):
    return True

def buildUploadedModel(model_type, state_dict_path):
    model = None
    if model_type == "resnet18":
        model = models.resnet18()
    
    if not model:
        raise TypeError("given model type did not match any of the options")
    
    try:
        model.load_state_dict(torch.load(state_dict_path))
    except RuntimeError as r:
        print(f'''Runtime error loading torch model from file:
{r}
Model is loaded from default values.
''')
    model.eval()
    return model
        

# Total variation regularization needs to be smaller on CIFAR-10:
# cfg.attack.regularization.total_variation.scale = 1e-3
# cfg.case.user.user_idx = 1