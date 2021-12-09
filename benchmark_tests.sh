# Run a suite of basic benchmarking tests for case 2

# Basic variants of current attack settings:
python benchmark_breaches.py name=invertinggradients_default case=2_single_imagenet attack=invertinggradients
python benchmark_breaches.py name=beyondinfering case=2_single_imagenet attack=beyondinfering
python benchmark_breaches.py name=deepleakage case=2_single_imagenet attack=deepleakage
python benchmark_breaches.py name=modern case=2_single_imagenet attack=modern
python benchmark_breaches.py name=seethroughgradients case=2_single_imagenet attack=seethroughgradients attack.regularization.deep_inversion=1e-4
python benchmark_breaches.py name=wei case=2_single_imagenet attack=wei

# A few small invertinggradients variants
python benchmark_breaches.py name=invertinggradients_unsigned case=2_single_imagenet attack=invertinggradients attack.optim.signed=False
python benchmark_breaches.py name=invertinggradients_double_opp case=2_single_imagenet attack=invertinggradients +attack.regularization.total_variation.double_opponents=True

python benchmark_breaches.py name=modern_signed case=2_single_imagenet attack=modern attack.optim.signed=True

python benchmark_breaches.py name=invertinggradients_angular case=2_single_imagenet attack=invertinggradients attack.objective.type=angular
python benchmark_breaches.py name=invertinggradients_l1 case=2_single_imagenet attack=invertinggradients attack.objective.type=l1
python benchmark_breaches.py name=invertinggradients_masked case=2_single_imagenet attack=invertinggradients attack.objective.type=masked-cosine-similarity
python benchmark_breaches.py name=invertinggradients_fast case=2_single_imagenet attack=invertinggradients attack.objective.type=fast-cosine-similarity

# Feature Reg back on the menu:
python benchmark_breaches.py name=invertinggradients_freg001 case=2_single_imagenet attack=invertinggradients +attack.regularization.features.scale=0.01
python benchmark_breaches.py name=invertinggradients_freg01 case=2_single_imagenet attack=invertinggradients +attack.regularization.features.scale=0.1
python benchmark_breaches.py name=invertinggradients_freg1 case=2_single_imagenet attack=invertinggradients +attack.regularization.features.scale=1.0
python benchmark_breaches.py name=invertinggradients_freg10 case=2_single_imagenet attack=invertinggradients +attack.regularization.features.scale=10
python benchmark_breaches.py name=invertinggradients_freg100 case=2_single_imagenet attack=invertinggradients +attack.regularization.features.scale=100


# Different inits for invertinggradients:
python benchmark_breaches.py name=invertinggradients_rand case=2_single_imagenet attack=invertinggradients attack.init=rand
python benchmark_breaches.py name=invertinggradients_zeros case=2_single_imagenet attack=invertinggradients attack.init=zeros
python benchmark_breaches.py name=invertinggradients_light case=2_single_imagenet attack=invertinggradients attack.init=light
python benchmark_breaches.py name=invertinggradients_red case=2_single_imagenet attack=invertinggradients attack.init=red
python benchmark_breaches.py name=invertinggradients_green case=2_single_imagenet attack=invertinggradients attack.init=green
python benchmark_breaches.py name=invertinggradients_blue case=2_single_imagenet attack=invertinggradients attack.init=blue
python benchmark_breaches.py name=invertinggradients_dark case=2_single_imagenet attack=invertinggradients attack.init=dark

python benchmark_breaches.py name=invertinggradients_redt case=2_single_imagenet attack=invertinggradients attack.init=red-true
python benchmark_breaches.py name=invertinggradients_greent case=2_single_imagenet attack=invertinggradients attack.init=green-true
python benchmark_breaches.py name=invertinggradients_bluet case=2_single_imagenet attack=invertinggradients attack.init=blue-true
python benchmark_breaches.py name=invertinggradients_darkt case=2_single_imagenet attack=invertinggradients attack.init=dark-true
python benchmark_breaches.py name=invertinggradients_lightt case=2_single_imagenet attack=invertinggradients attack.init=light-true

python benchmark_breaches.py name=invertinggradients_p4 case=2_single_imagenet attack=invertinggradients attack.init=patterned-4
python benchmark_breaches.py name=invertinggradients_p8 case=2_single_imagenet attack=invertinggradients attack.init=patterned-8
python benchmark_breaches.py name=invertinggradients_p16 case=2_single_imagenet attack=invertinggradients attack.init=patterned-16
python benchmark_breaches.py name=invertinggradients_p32 case=2_single_imagenet attack=invertinggradients attack.init=patterned-32
python benchmark_breaches.py name=invertinggradients_p64 case=2_single_imagenet attack=invertinggradients attack.init=patterned-64
python benchmark_breaches.py name=invertinggradients_p128 case=2_single_imagenet attack=invertinggradients attack.init=patterned-128
