# feat estimation with different batch sizes
python classattack_feat_one_shot.py attack=clsattack case.server.name=class_malicious_parameters case/data=ImageNet case.user.provide_labels=True base_dir=/cmlscratch/ywen/breaching/breaching/outputs num_trials=50 +num_to_est=900 +batch_size=4