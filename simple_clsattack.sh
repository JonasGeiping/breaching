# no cls collision
python classattack_breaches.py name=clsattack_no_collision_bsize1 case.user.user_idx=0 case.user.num_data_points=1 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True
python classattack_breaches.py name=clsattack_no_collision_bsize4 case.user.user_idx=0 case.user.num_data_points=4 attack=clsattack case.server.name=class_malicious_parameters case.user.provide_labels=True
