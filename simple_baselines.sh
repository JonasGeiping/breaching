# Literally the easiest baselines to attack. Check some scores and report values.
# This is a test ablation.
# A better ablation would have a cooler dataset and multiple datapoints to be checked.

python breach.py name=ig cfg.user.data_idx=0

# IG ablation stuff:
python breach.py name=ig_fewer_steps cfg.user.data_idx=0 cfg.attack.optim.max_iterations=10_000
python breach.py name=ig_trials cfg.user.data_idx=0 cfg.attack.restarts.num_trials=8
python breach.py name=ig_unsigned cfg.user.data_idx=0 cfg.attack.optim.signed=False
python breach.py name=ig_warmup cfg.user.data_idx=0 cfg.attack.optim.warmup=50
python breach.py name=ig_large_step cfg.user.data_idx=0 cfg.attack.optim.step_size=0.5
python breach.py name=ig_langevin cfg.user.data_idx=0 cfg.attack.optim.langevin_noise=0.2
python breach.py name=ig_iso cfg.user.data_idx=0 cfg.attack.regularization.total_variation.inner_exp=2 cfg.attack.regularization.total_variation.outer_exp=0.5
python breach.py name=ig_euclidean cfg.user.data_idx=0 cfg.attack.objective=euclidean
python breach.py name=ig_euclidean_unsigned cfg.user.data_idx=0 cfg.attack.optim.signed=False cfg.attack.objective=euclidean

# Other attacks:
python breach.py name=deepleakage cfg.user.data_idx=0 attack=deepleakage
python breach.py name=beyond cfg.user.data_idx=0 attack=beyondinfering
python breach.py name=seethrough_kindof cfg.user.data_idx=0 attack=seethroughgradients
