# # no cls collision
# python classattack_breaches.py name=clsattack_no_collision_bsize1 case.user.user_idx=0 case.user.num_data_points=1 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=unique-class
# python classattack_breaches.py name=clsattack_no_collision_bsize4 case.user.user_idx=0 case.user.num_data_points=4 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=balanced num_trials=50
#
# # Larger batches:
# python classattack_breaches.py name=clsattack_no_collision_bsize8 case.user.user_idx=0 case.user.num_data_points=8 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=balanced num_trials=50
#
# # Larger batch sizes with random partitions:
# # Limit testing the code here:
# python classattack_breaches.py name=clsattack_no_collision_bsize64_rand case.user.user_idx=0 case.user.num_data_points=64 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python classattack_breaches.py name=clsattack_no_collision_bsize128_rand case.user.user_idx=0 case.user.num_data_points=128 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python classattack_breaches.py name=clsattack_no_collision_bsize256_rand case.user.user_idx=0 case.user.num_data_points=256 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python classattack_breaches.py name=clsattack_no_collision_bsize512_rand case.user.user_idx=0 case.user.num_data_points=512 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=50
# python classattack_breaches.py name=clsattack_no_collision_bsize1024_rand case.user.user_idx=0 case.user.num_data_points=1024 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=25
#
#
# # cls collision
# python classattack_breaches.py name=clsattack_mixup_bsize4_freq2 case.user.user_idx=0 case.user.num_data_points=4 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=mixup +case.data.mixup_freq=2 case/data=ImageNet case.data.default_clients=100
# python classattack_breaches.py name=clsattack_mixup_bsize8_freq4 case.user.user_idx=0 case.user.num_data_points=8 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=mixup +case.data.mixup_freq=4 case/data=ImageNet case.data.default_clients=100
# python classattack_breaches.py name=clsattack_mixup_bsize16_freq8 case.user.user_idx=0 case.user.num_data_points=16 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=mixup +case.data.mixup_freq=8 case/data=ImageNet case.data.default_clients=100
# python classattack_breaches.py name=clsattack_mixup_bsize32_freq16 case.user.user_idx=0 case.user.num_data_points=32 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=mixup +case.data.mixup_freq=16 case/data=ImageNet case.data.default_clients=100


# Run binary attack / cls collision in a variety of settings:
# 1) random batches
python classattack_breaches.py name=clsattack_b01_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=1 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
python classattack_breaches.py name=clsattack_b32_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
python classattack_breaches.py name=clsattack_b64_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=64 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
python classattack_breaches.py name=clsattack_b64_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=64 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
python classattack_breaches.py name=clsattack_b128_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=128 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
python classattack_breaches.py name=clsattack_b256_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=256 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
python classattack_breaches.py name=clsattack_b500_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=500 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True

# 2) batches with a single class (the worst case for the binary attack)
python classattack_breaches.py name=clsattack_b1_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=1 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
python classattack_breaches.py name=clsattack_b4_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=4 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
python classattack_breaches.py name=clsattack_b8_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=8 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
python classattack_breaches.py name=clsattack_b16_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=16 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
python classattack_breaches.py name=clsattack_b32_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
python classattack_breaches.py name=clsattack_b50_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=50 attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
#
# # Compare to honest methods:
# # IG
# python benchmark_breaches.py name=igattack_b1_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=1 attack=invertinggradients case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=igattack_b4_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=4 attack=invertinggradients case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=igattack_b8_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=8 attack=invertinggradients case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=igattack_b16_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=16 attack=invertinggradients case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=igattack_b32_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=invertinggradients  case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=igattack_b50_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=50 attack=invertinggradients  case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
#
# python benchmark_breaches.py name=igattack_b1_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=1 attack=invertinggradients case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=igattack_b4_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=4 attack=invertinggradients case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=igattack_b8_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=8 attack=invertinggradients case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=igattack_b16_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=16 attack=invertinggradients case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=igattack_b32_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=invertinggradients  case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=igattack_b50_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=50 attack=invertinggradients  case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# # Yin
# python benchmark_breaches.py name=yinattack_b1_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=1 attack=seethroughgradients case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 attack.regularization.deep_inversion.scale=1e-4
# python benchmark_breaches.py name=yinattack_b4_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=4 attack=seethroughgradients case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 attack.regularization.deep_inversion.scale=1e-4
# python benchmark_breaches.py name=yinattack_b8_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=8 attack=seethroughgradients case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 attack.regularization.deep_inversion.scale=1e-4
# python benchmark_breaches.py name=yinattack_b16_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=16 attack=seethroughgradients case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 attack.regularization.deep_inversion.scale=1e-4
# python benchmark_breaches.py name=yinattack_b32_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=seethroughgradients  case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 attack.regularization.deep_inversion.scale=1e-4
# python benchmark_breaches.py name=yinattack_b50_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=50 attack=seethroughgradients  case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100 attack.regularization.deep_inversion.scale=1e-4
#
# python benchmark_breaches.py name=yinattack_b1_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=1 attack=seethroughgradients case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 attack.regularization.deep_inversion.scale=1e-4
# python benchmark_breaches.py name=igsattack_b4_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=4 attack=seethroughgradients case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 attack.regularization.deep_inversion.scale=1e-4
# python benchmark_breaches.py name=yinattack_b8_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=8 attack=seethroughgradients case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 attack.regularization.deep_inversion.scale=1e-4
# python benchmark_breaches.py name=yinattack_b16_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=16 attack=seethroughgradients case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 attack.regularization.deep_inversion.scale=1e-4
# python benchmark_breaches.py name=yinattack_b32_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=seethroughgradients  case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 attack.regularization.deep_inversion.scale=1e-4
# python benchmark_breaches.py name=yinattack_b50_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=50 attack=seethroughgradients  case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100 attack.regularization.deep_inversion.scale=1e-4
#
# # modern
# python benchmark_breaches.py name=modernattack_b1_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=1 attack=modern case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=modernattack_b4_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=4 attack=modern case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=modernattack_b8_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=8 attack=modern case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=modernattack_b16_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=16 attack=modern case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=modernattack_b32_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=modern  case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=modernattack_b50_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=50 attack=modern  case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
#
# python benchmark_breaches.py name=modernattack_b1_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=1 attack=modern case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=modernattack_b4_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=4 attack=modern case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=modernattack_b8_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=8 attack=modern case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=modernattack_b16_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=16 attack=modern case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=modernattack_b32_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=modern  case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=modernattack_b50_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=50 attack=modern  case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
#
# # clsattack hyperparams
# python benchmark_breaches.py name=clsparamsattack_b1_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=1 attack=clsattack case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=clsparamsattack_b4_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=4 attack=clsattack case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=clsparamsattack_b8_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=8 attack=clsattack case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=clsparamsattack_b16_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=16 attack=clsattack case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=clsparamsattack_b32_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=clsattack  case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
# python benchmark_breaches.py name=clsparamsattack_b50_rand_fullImageNet case.user.user_idx=0 case.user.num_data_points=50 attack=clsattack  case/data=ImageNet case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
#
# python benchmark_breaches.py name=clsparamsattack_b1_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=1 attack=clsattack case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=modernsattack_b4_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=4 attack=clsattack case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=clsparamsattack_b8_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=8 attack=clsattack case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=clsparamsattack_b16_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=16 attack=clsattack case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=clsparamsattack_b32_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=32 attack=clsattack  case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100
# python benchmark_breaches.py name=clsparamsattack_b50_unique_fullImageNet case.user.user_idx=0 case.user.num_data_points=50 attack=clsattack  case/data=ImageNet case.user.provide_labels=True case.data.partition=unique-class case.data.default_clients=100