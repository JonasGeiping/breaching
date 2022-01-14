# no cls collision
python classattack_breaches.py name=clsattack_no_collision_bsize1 case.user.user_idx=0 case.user.num_data_points=1 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=unique-class
python classattack_breaches.py name=clsattack_no_collision_bsize4 case.user.user_idx=0 case.user.num_data_points=4 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=balanced num_trials=50

# Larger batches:
python classattack_breaches.py name=clsattack_no_collision_bsize8 case.user.user_idx=0 case.user.num_data_points=4 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=balanced num_trials=50
python classattack_breaches.py name=clsattack_no_collision_bsize16 case.user.user_idx=0 case.user.num_data_points=16 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=balanced num_trials=50
python classattack_breaches.py name=clsattack_no_collision_bsize32 case.user.user_idx=0 case.user.num_data_points=32 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=balanced num_trials=50
python classattack_breaches.py name=clsattack_no_collision_bsize64 case.user.user_idx=0 case.user.num_data_points=64 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=balanced num_trials=50
python classattack_breaches.py name=clsattack_no_collision_bsize128 case.user.user_idx=0 case.user.num_data_points=128 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=balanced num_trials=50





# Larger batch sizes with random partitions:
# Limit testing the code here:
python classattack_breaches.py name=clsattack_no_collision_bsize64 case.user.user_idx=0 case.user.num_data_points=64 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
python classattack_breaches.py name=clsattack_no_collision_bsize128 case.user.user_idx=0 case.user.num_data_points=128 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
python classattack_breaches.py name=clsattack_no_collision_bsize256 case.user.user_idx=0 case.user.num_data_points=256 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
python classattack_breaches.py name=clsattack_no_collision_bsize512 case.user.user_idx=0 case.user.num_data_points=512 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
python classattack_breaches.py name=clsattack_no_collision_bsize1024 case.user.user_idx=0 case.user.num_data_points=1024 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True case.data.partition=random case.data.default_clients=100
