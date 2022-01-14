import torch

def feature_distribution(model, server, measurement):
    """Compute the mean and std of the feature layer of the given network."""
    features = dict()
    #setup = dict(device=device)
    def named_hook(name):
        def hook_fn(module, input, output):
            features[name] = input[0]
        return hook_fn

    name = 'linear1'
    module = model.transformer_encoder.layers[0].linear1
    hook = module.register_forward_hook(named_hook(name))
    feature_layer_name = name
    
    feats = []
    feats_before = []
    model.train()
    model.to(**server.setup)
    print(f'Computing feature distribution before the {feature_layer_name} layer from external data.')
    for i, batch in enumerate(server.external_dataloader):
        inputs = batch['input_ids'].to(device=server.setup['device'])
        model(inputs)
        feats.append(features[name].detach().view(inputs.shape[0]*inputs.shape[1], -1).clone().cpu())
        
    std, mu = torch.std_mean(torch.mm(torch.cat(feats), measurement.unsqueeze(1)).squeeze())
    model.eval()
    model.cpu()
    hook.remove()
    print(f'Feature mean is {mu.item()}, feature std is {std.item()}.')
    return std, mu


def set_MHA(model, server, sequence_token_weight=0.075, pos=0):
    # Let's set the query matrix to produce just the first positional encoding (or could be any index - might want last index)
    qkv_shape = getattr(model.transformer_encoder.layers, '0').self_attn.in_proj_weight.data.shape[0] 

    dummy_data = next(iter(server.external_dataloader))['input_ids'].to(device=server.setup['device'])
    just_pos = torch.stack([getattr(model.transformer_encoder.layers, '0').norm1(model.pos_encoder(torch.zeros_like(dummy_data)) * math.sqrt(model.d_model))]).cpu().squeeze()

    # Q matrix setup
    # We make the weight 0, and the bias some (large multiple of) positional encoding
    # Only coded here for one MHA layer at the beginning of the model... 
    # Make the position super super large to skew softmax
    getattr(model.transformer_encoder.layers, '0').self_attn.in_proj_bias.data[:qkv_shape//3] = 1000*just_pos[0,pos,:]
    getattr(model.transformer_encoder.layers, '0').self_attn.in_proj_weight.data[:qkv_shape//3] = torch.zeros((qkv_shape//3, qkv_shape//3))

    # K matrix setup (identity)
    getattr(model.transformer_encoder.layers, '0').self_attn.in_proj_weight.data[qkv_shape//3:2*(qkv_shape//3)] = torch.eye(qkv_shape//3)

    # V matrix setup (identity)
    getattr(model.transformer_encoder.layers, '0').self_attn.in_proj_weight.data[2*(qkv_shape//3):] = torch.eye(qkv_shape//3)

    # So, (QK^T)V just adds the same vector (first word embedding) to each word in the sequence.  

    # Linear layer at the end of MHA - set to small value to not 'skew' embeddings too much
    getattr(model.transformer_encoder.layers, '0').self_attn.out_proj.weight.data = sequence_token_weight*torch.eye(qkv_shape//3)
    

def make_imprint_layer(model, measurement, mean, std):
    '''
    measurement is the Gaussian vector we take inner product w.r.t.
    mean, std = mean, std of features from feature_distribution
    '''
    from statistics import NormalDist
    
    def _get_bins(mean, std, num_bins):
        bins = []
        mass_per_bin = 1 / (num_bins)
        bins.append(-10)  # -Inf is not great here, but NormalDist(mu=0, sigma=1).cdf(10) approx 1
        for i in range(1, num_bins):
            bins.append(NormalDist().inv_cdf(i * mass_per_bin)*std + mean)
        return bins

    def _make_biases(bias_layer, bins):
        new_biases = torch.zeros_like(bias_layer.data)
        for i in range(new_biases.shape[0]):
            new_biases[i] = -bins[i]
        return new_biases

    bins = _get_bins(mean, std, model.d_model)
    getattr(model.transformer_encoder.layers, '0').linear1.weight.data = measurement.repeat(model.d_model, 1)
    getattr(model.transformer_encoder.layers, '0').linear1.bias.data = _make_biases(getattr(model.transformer_encoder.layers, '0').linear1.bias, bins)

