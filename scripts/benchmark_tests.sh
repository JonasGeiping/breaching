# # Run a suite of basic benchmarking tests for case 2
#
# # Basic variants of current attack settings:
# python benchmark_breaches.py name=invertinggradients_default case=2_single_imagenet attack=invertinggradients
# python benchmark_breaches.py name=beyondinfering case=2_single_imagenet attack=beyondinfering
# python benchmark_breaches.py name=deepleakage case=2_single_imagenet attack=deepleakage
# python benchmark_breaches.py name=modern case=2_single_imagenet attack=modern
# python benchmark_breaches.py name=seethroughgradients case=2_single_imagenet attack=seethroughgradients attack.regularization.deep_inversion=1e-4
# python benchmark_breaches.py name=wei case=2_single_imagenet attack=wei
#
# # A few small invertinggradients variants
# python benchmark_breaches.py name=invertinggradients_unsigned case=2_single_imagenet attack=invertinggradients attack.optim.signed=False
# python benchmark_breaches.py name=invertinggradients_double_opp case=2_single_imagenet attack=invertinggradients +attack.regularization.total_variation.double_opponents=True
#
# python benchmark_breaches.py name=modern_signed case=2_single_imagenet attack=modern attack.optim.signed=True
#
# python benchmark_breaches.py name=invertinggradients_angular case=2_single_imagenet attack=invertinggradients attack.objective.type=angular
# python benchmark_breaches.py name=invertinggradients_l1 case=2_single_imagenet attack=invertinggradients attack.objective.type=l1
# python benchmark_breaches.py name=invertinggradients_masked case=2_single_imagenet attack=invertinggradients attack.objective.type=masked-cosine-similarity
# python benchmark_breaches.py name=invertinggradients_fast case=2_single_imagenet attack=invertinggradients attack.objective.type=fast-cosine-similarity
#
# # Feature Reg back on the menu:
# python benchmark_breaches.py name=invertinggradients_freg001 case=2_single_imagenet attack=invertinggradients +attack.regularization.features.scale=0.01
# python benchmark_breaches.py name=invertinggradients_freg01 case=2_single_imagenet attack=invertinggradients +attack.regularization.features.scale=0.1
# python benchmark_breaches.py name=invertinggradients_freg1 case=2_single_imagenet attack=invertinggradients +attack.regularization.features.scale=1.0
# python benchmark_breaches.py name=invertinggradients_freg10 case=2_single_imagenet attack=invertinggradients +attack.regularization.features.scale=10
# python benchmark_breaches.py name=invertinggradients_freg100 case=2_single_imagenet attack=invertinggradients +attack.regularization.features.scale=100
#
#
# # Different inits for invertinggradients:
# python benchmark_breaches.py name=invertinggradients_rand case=2_single_imagenet attack=invertinggradients attack.init=rand
# python benchmark_breaches.py name=invertinggradients_zeros case=2_single_imagenet attack=invertinggradients attack.init=zeros
# python benchmark_breaches.py name=invertinggradients_light case=2_single_imagenet attack=invertinggradients attack.init=light
# python benchmark_breaches.py name=invertinggradients_red case=2_single_imagenet attack=invertinggradients attack.init=red
# python benchmark_breaches.py name=invertinggradients_green case=2_single_imagenet attack=invertinggradients attack.init=green
# python benchmark_breaches.py name=invertinggradients_blue case=2_single_imagenet attack=invertinggradients attack.init=blue
# python benchmark_breaches.py name=invertinggradients_dark case=2_single_imagenet attack=invertinggradients attack.init=dark
#
# python benchmark_breaches.py name=invertinggradients_redt case=2_single_imagenet attack=invertinggradients attack.init=red-true
# python benchmark_breaches.py name=invertinggradients_greent case=2_single_imagenet attack=invertinggradients attack.init=green-true
# python benchmark_breaches.py name=invertinggradients_bluet case=2_single_imagenet attack=invertinggradients attack.init=blue-true
# python benchmark_breaches.py name=invertinggradients_darkt case=2_single_imagenet attack=invertinggradients attack.init=dark-true
# python benchmark_breaches.py name=invertinggradients_lightt case=2_single_imagenet attack=invertinggradients attack.init=light-true
#
# python benchmark_breaches.py name=invertinggradients_p4 case=2_single_imagenet attack=invertinggradients attack.init=patterned-4
# python benchmark_breaches.py name=invertinggradients_p8 case=2_single_imagenet attack=invertinggradients attack.init=patterned-8
# python benchmark_breaches.py name=invertinggradients_p16 case=2_single_imagenet attack=invertinggradients attack.init=patterned-16
# python benchmark_breaches.py name=invertinggradients_p32 case=2_single_imagenet attack=invertinggradients attack.init=patterned-32
# python benchmark_breaches.py name=invertinggradients_p64 case=2_single_imagenet attack=invertinggradients attack.init=patterned-64
# python benchmark_breaches.py name=invertinggradients_p128 case=2_single_imagenet attack=invertinggradients attack.init=patterned-128

# python benchmark_breaches.py name=invertinggradients_soft_sign case=2_single_imagenet attack=invertinggradients attack.optim.signed="soft"
# python benchmark_breaches.py name=invertinggradients_double_opp_soft_sign case=2_single_imagenet attack=invertinggradients attack.optim.signed="soft" +attack.regularization.total_variation.double_opponents=True
#
# python benchmark_breaches.py name=modern2_freg01 case=2_single_imagenet attack=modern2
# python benchmark_breaches.py name=modern2_freg001 case=2_single_imagenet attack=modern2 attack.regularization.features.scale=0.01
# python benchmark_breaches.py name=modern2_freg1 case=2_single_imagenet attack=modern2 attack.regularization.features.scale=1.0
#
# python benchmark_breaches.py name=modern2_l1tv case=2_single_imagenet attack=modern2 attack.regularization.total_variation.inner_exp=1 attack.regularization.total_variation.outer_exp=1
#
# # Benchmark some data augmentations:
# python benchmark_breaches.py name=invertinggradients_shift1 case=2_single_imagenet attack=invertinggradients +attack.augmentations.continuous_shift.shift=1.0
# python benchmark_breaches.py name=invertinggradients_shift01 case=2_single_imagenet attack=invertinggradients +attack.augmentations.continuous_shift.shift=0.1
# python benchmark_breaches.py name=invertinggradients_shift05 case=2_single_imagenet attack=invertinggradients +attack.augmentations.continuous_shift.shift=0.5
# python benchmark_breaches.py name=invertinggradients_shift2 case=2_single_imagenet attack=invertinggradients +attack.augmentations.continuous_shift.shift=2.0
# python benchmark_breaches.py name=invertinggradients_shift10 case=2_single_imagenet attack=invertinggradients +attack.augmentations.continuous_shift.shift=10
# python benchmark_breaches.py name=invertinggradients_shift50 case=2_single_imagenet attack=invertinggradients +attack.augmentations.continuous_shift.shift=50 +attack.augmentations.continuous_shift.padding=circular
#
# python benchmark_breaches.py name=invertinggradients_colorjitter case=2_single_imagenet attack=invertinggradients +attack.augmentations.colorjitter={}
#
# python benchmark_breaches.py name=invertinggradients_shift1_nodiff case=2_single_imagenet attack=invertinggradients +attack.augmentations.continuous_shift.shift=1.0 attack.differentiable_augmentations=False
# python benchmark_breaches.py name=invertinggradients_flip case=2_single_imagenet attack=invertinggradients +attack.augmentations.flip={}
# python benchmark_breaches.py name=invertinggradients_flip_nodiff case=2_single_imagenet attack=invertinggradients +attack.augmentations.flip={} attack.differentiable_augmentations=False
#
#
# python benchmark_breaches.py name=invertinggradients_median case=2_single_imagenet attack=invertinggradients +attack.augmentations.median={}
# python benchmark_breaches.py name=invertinggradients_median_nodiff case=2_single_imagenet attack=invertinggradients +attack.augmentations.median={} attack.differentiable_augmentations=False
#
#
# python benchmark_breaches.py name=invertinggradients_aa3 case=2_single_imagenet attack=invertinggradients +attack.augmentations.antialias.width=3
# python benchmark_breaches.py name=invertinggradients_aa3_nodiff case=2_single_imagenet attack=invertinggradients +attack.augmentations.antialias.width3 attack.differentiable_augmentations=False
#
#
# # Misc:
# python benchmark_breaches.py name=invertinggradients_adamsafe case=2_single_imagenet attack=invertinggradients attack.optim.optimizer=adam-safe
#
# python benchmark_breaches.py name=invertinggradients_angular_unsigned case=2_single_imagenet attack=invertinggradients attack.objective.type=angular attack.optim.signed=none
# python benchmark_breaches.py name=invertinggradients_angular_unsigned_lr001 case=2_single_imagenet attack=invertinggradients attack.objective.type=angular attack.optim.signed=none attack.optim.step_size=0.01
# python benchmark_breaches.py name=invertinggradients_angular_unsigned_lr1 case=2_single_imagenet attack=invertinggradients attack.objective.type=angular attack.optim.signed=none attack.optim.step_size=1.0
#
# python benchmark_breaches.py name=invertinggradients_angular_lr1 case=2_single_imagenet attack=invertinggradients attack.objective.type=angular attack.optim.step_size=0.01
# python benchmark_breaches.py name=invertinggradients_angular_lr001 case=2_single_imagenet attack=invertinggradients attack.objective.type=angular attack.optim.step_size=1.0

# New suggestions:
python benchmark_breaches.py name=new_modern case=2_single_imagenet attack=modern
python benchmark_breaches.py name=new_modern_w50 case=2_single_imagenet attack=modern attack.optim.warmup=50
python benchmark_breaches.py name=new_modern_w250 case=2_single_imagenet attack=modern attack.optim.warmup=250
python benchmark_breaches.py name=new_modern_w50_s001 case=2_single_imagenet attack=modern attack.optim.warmup=50 attack.optim.step_size=0.01
python benchmark_breaches.py name=new_modern_p4 case=2_single_imagenet attack=modern attack.init=patterned-4
python benchmark_breaches.py name=new_modern_p4 case=2_single_imagenet attack=modern attack.init=patterned-8
python benchmark_breaches.py name=new_modern_p4_w50 case=2_single_imagenet attack=modern attack.init=patterned-4 attack.optim.warmup=50

python benchmark_breaches.py name=new_modern_p4_w50_nofeat case=2_single_imagenet attack=modern attack.init=patterned-4 attack.optim.warmup=50 attack.regularization.features.scale=0
python benchmark_breaches.py name=new_modern_p4_w50_nodeepinverse case=2_single_imagenet attack=modern attack.init=patterned-4 attack.optim.warmup=50 attack.regularization.deep_inversion.scale=0
