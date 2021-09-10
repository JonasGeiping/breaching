# Literally the easiest baselines to attack. Check some scores and report values.
# This is a test ablation.
# A better ablation would have a cooler dataset and multiple datapoints to be checked.


# IG ablation stuff:
python breach.py name=ig_fewer_steps case.user.data_idx=0 case.model=resnet50 attack.optim.max_iterations=10_000
python breach.py name=ig_trials case.user.data_idx=0 case.model=resnet50 attack.restarts.num_trials=8
python breach.py name=ig_unsigned case.user.data_idx=0 case.model=resnet50 attack.optim.signed=False
python breach.py name=ig_warmup case.user.data_idx=0 case.model=resnet50 attack.optim.warmup=50
python breach.py name=ig_large_step case.user.data_idx=0 case.model=resnet50 attack.optim.step_size=0.5
python breach.py name=ig_langevin case.user.data_idx=0 case.model=resnet50 attack.optim.langevin_noise=0.2
python breach.py name=ig_iso case.user.data_idx=0 case.model=resnet50 attack.regularization.total_variation.inner_exp=2 attack.regularization.total_variation.outer_exp=0.5
python breach.py name=ig_euclidean case.user.data_idx=0 case.model=resnet50 attack.objective.type=euclidean
python breach.py name=ig_euclidean_unsigned case.user.data_idx=0 case.model=resnet50 attack.optim.signed=False attack.objective.type=euclidean

# Other attacks:
python breach.py name=ig case.user.data_idx=0 case.model=resnet50
python breach.py name=deepleakage case.user.data_idx=0 case.model=resnet50 attack=deepleakage
python breach.py name=beyond case.user.data_idx=0 case.model=resnet50 attack=beyondinfering
python breach.py name=seethrough_kindof case.user.data_idx=0 case.model=resnet50 attack=seethroughgradients
python breach.py name=modern case.user.data_idx=0 case.model=resnet50 attack=modern

# modern ablations:
python breach.py name=modern_notv case.user.data_idx=0 case.model=resnet50 attack=modern attack.regularization.total_variation.scale=0.0
python breach.py name=modern_nol2 case.user.data_idx=0 case.model=resnet50 attack=modern attack.regularization.norm.scale=0.0
python breach.py name=modern_nodi case.user.data_idx=0 case.model=resnet50 attack=modern attack.regularization.deep_inversion.scale=0.0
python breach.py name=modern_di10 case.user.data_idx=0 case.model=resnet50 attack=modern attack.regularization.deep_inversion.scale=0.1
python breach.py name=modern_di100 case.user.data_idx=0 case.model=resnet50 attack=modern attack.regularization.deep_inversion.scale=1.0

python breach.py name=modern_ll case.user.data_idx=0 case.model=resnet50 attack=modern attack.optim.langevin_noise=0.0
python breach.py name=modern_l01 case.user.data_idx=0 case.model=resnet50 attack=modern attack.optim.langevin_noise=0.1
python breach.py name=modern_l001 case.user.data_idx=0 case.model=resnet50 attack=modern attack.optim.langevin_noise=0.01
python breach.py name=modern_nw case.user.data_idx=0 case.model=resnet50 attack=modern attack.optim.warmup=0

# Other inits:
python breach.py name=modern_moco case.user.data_idx=0 case.model=resnet50 attack=modern case.server.model_state=moco
python breach.py name=ig_moco case.user.data_idx=0 case.model=resnet50 case.server.model_state=moco

python breach.py name=modern_untrained case.user.data_idx=0 case.model=resnet50 attack=modern case.server.model_state=untrained
python breach.py name=ig_untrained case.user.data_idx=0 case.model=resnet50 case.server.model_state=untrained

python breach.py name=modern_untrained4 case.user.data_idx=0 case.model=resnet50 attack=modern case.server.model_state=untrained case.num_queries=4
python breach.py name=ig_untrained4 case.user.data_idx=0 case.model=resnet50 case.server.model_state=untrained case.num_queries=4

python breach.py name=modern_SSL case.user.data_idx=0 case.model=resnet50ssl attack=modern
python breach.py name=ig_SSL case.user.data_idx=0 case.model=resnet50ssl

python breach.py name=modern_SWSL case.user.data_idx=0 case.model=resnet50swsl attack=modern
python breach.py name=ig_SWSL case.user.data_idx=0 case.model=resnet50swsl


python breach.py name=modern_ortho case.user.data_idx=0 case.model=resnet50 attack=modern case.server.model_state=orthogonal
python breach.py name=ig_ortho case.user.data_idx=0 case.model=resnet50 case.server.model_state=orthogonal

python breach.py name=modern_ortho4 case.user.data_idx=0 case.model=resnet50 attack=modern case.server.model_state=orthogonal case.num_queries=4
python breach.py name=ig_ortho4 case.user.data_idx=0 case.model=resnet50 case.server.model_state=orthogonal case.num_queries=4
