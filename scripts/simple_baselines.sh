# Literally the easiest baselines to attack. Check some scores and report values.
# This is a test ablation.
# A better ablation would have a cooler dataset and multiple datapoints to be checked.

python breach.py name=ig case.user.data_idx=0

# IG ablation stuff:
python breach.py name=ig_fewer_steps case.user.data_idx=0 attack.optim.max_iterations=10_000
python breach.py name=ig_trials case.user.data_idx=0 attack.restarts.num_trials=8
python breach.py name=ig_unsigned case.user.data_idx=0 attack.optim.signed=False
python breach.py name=ig_warmup case.user.data_idx=0 attack.optim.warmup=50
python breach.py name=ig_large_step case.user.data_idx=0 attack.optim.step_size=0.5
python breach.py name=ig_langevin case.user.data_idx=0 attack.optim.langevin_noise=0.2
python breach.py name=ig_iso case.user.data_idx=0 attack.regularization.total_variation.inner_exp=2 attack.regularization.total_variation.outer_exp=0.5
python breach.py name=ig_euclidean case.user.data_idx=0 attack.objective=euclidean
python breach.py name=ig_euclidean_unsigned case.user.data_idx=0 attack.optim.signed=False attack.objective=euclidean

# Other attacks:
python breach.py name=deepleakage case.user.data_idx=0 attack=deepleakage
python breach.py name=beyond case.user.data_idx=0 attack=beyondinfering
python breach.py name=seethrough_kindof case.user.data_idx=0 attack=seethroughgradients
