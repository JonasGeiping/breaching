# Run a suite of basic benchmarking tests for case 2

# Basic variants of current attack settings:
python benchmark_breaches.py name=invertinggradients_default case=2_single_imagenet attack=invertinggradients
python benchmark_breaches.py name=beyondinfering case=2_single_imagenet attack=beyondinfering
python benchmark_breaches.py name=deepleakage case=2_single_imagenet attack=deepleakage
python benchmark_breaches.py name=modern case=2_single_imagenet attack=modern
python benchmark_breaches.py name=seethroughgradients case=2_single_imagenet attack=seethroughgradients
python benchmark_breaches.py name=wei case=2_single_imagenet attack=wei

# A few small invertinggradients variants
python benchmark_breaches.py name=invertinggradients_unsigned case=2_single_imagenet attack=invertinggradients attack.optim.signed=False
python benchmark_breaches.py name=invertinggradients_double_opp case=2_single_imagenet attack=invertinggradients +attack.regularization.total_variation.double_opponents=True
