import torch
import breaching


cfg = breaching.get_config(overrides=["case=4_fedavg_small_scale", "attack=invertinggradients_test", "case/data=CIFAR10"])

device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

print(setup)

# Total variation regularization needs to be smaller on CIFAR-10:
cfg.attack.regularization.total_variation.scale = 1e-3
# cfg.case.user.user_idx = 1


# print(cfg)

user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)

server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)

user.plot(true_user_data, saveFile="true_data")

print("reconstructing...")
reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)


# metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload], 
#                                     server.model, order_batch=True, compute_full_iip=False, 
#                                     cfg_case=cfg.case, setup=setup)

print("done")
user.plot(reconstructed_user_data, saveFile="reconstructed_data")