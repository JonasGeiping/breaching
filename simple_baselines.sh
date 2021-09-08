# Literally the easiest baselines to attack. Check some scores and report values.
# This is a test ablation.
# A better ablation would have a cooler dataset and multiple datapoints to be checked.

python breach.py name=ig user.data_idx=0

# IG ablation stuff:
python breach.py name=ig_fewer_steps user.data_idx=0 attack.optim.max_iterations=10_000
python breach.py name=ig_trials user.data_idx=0 attack.restarts.num_trials=8
python breach.py name=ig_unsigned user.data_idx=0 attack.optim.signed=False
python breach.py name=ig_warmup user.data_idx=0 attack.optim.warmup=50
python breach.py name=ig_large_step user.data_idx=0 attack.optim.step_size=0.5
python breach.py name=ig_langevin user.data_idx=0 attack.optim.langevin_noise=0.2
python breach.py name=ig_iso user.data_idx=0 attack.regularization.total_variation.inner_exp=2 attack.regularization.total_variation.outer_exp=0.5
python breach.py name=ig_euclidean user.data_idx=0 attack.objective=euclidean
python breach.py name=ig_euclidean_unsigned user.data_idx=0 attack.optim.signed=False attack.objective=euclidean

# Other attacks:
python breach.py name=deepleakage user.data_idx=0 attack=deepleakage
python breach.py name=beyond user.data_idx=0 attack=beyondinfering
python breach.py name=seethrough_kindof user.data_idx=0 attack=seethroughgradients
