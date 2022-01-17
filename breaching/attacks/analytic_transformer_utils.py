import torch
import numpy as np
from statistics import NormalDist
from scipy.optimize import linear_sum_assignment # Better than greedy search? 
from collections import defaultdict
import math 

def feature_distribution(model, server, measurement, block_num=0):
    """Compute the mean and std of the feature layer of the given network."""
    features = dict()
    #setup = dict(device=device)
    def named_hook(name):
        def hook_fn(module, input, output):
            features[name] = input[0]
        return hook_fn

    name = 'linear1'
    module = model.transformer_encoder.layers[block_num].linear1
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


def set_MHA(model, server, sequence_token_weight=100.0, pos=0, attention_block=0, v_proportion=1.0):
    # Let's set the query matrix to produce just the first positional encoding (or could be any index - might want last index)
    model.transformer_encoder.layers[attention_block].self_attn.in_proj_weight.data.shape[0] 

    dummy_data = next(iter(server.external_dataloader))['input_ids']
    just_pos =torch.stack([model.transformer_encoder.layers[0].norm1(
        model.pos_encoder(model.encoder(torch.zeros_like(dummy_data))))]).cpu().squeeze()
    print(dummy_data.shape)
    print(just_pos.shape)

    # Q matrix setup
    # We make the weight 0, and the bias some (large multiple of) positional encoding
    # Only coded here for one MHA layer at the beginning of the model... 
    # Make the position super super large to skew softmax
    qkv_shape = model.transformer_encoder.layers[attention_block].self_attn.in_proj_weight.data.shape[0] 
    model.transformer_encoder.layers[attention_block].self_attn.in_proj_bias.data[:qkv_shape//3] = 1000000*just_pos[0,pos,:]
    model.transformer_encoder.layers[attention_block].self_attn.in_proj_weight.data[:qkv_shape//3] = torch.zeros((qkv_shape//3, qkv_shape//3))

    # K matrix setup (identity)
    model.transformer_encoder.layers[attention_block].self_attn.in_proj_weight.data[qkv_shape//3:2*(qkv_shape//3)] = torch.eye(qkv_shape//3)

    # V matrix setup (truncated identity)
    v_data = torch.eye(qkv_shape//3)
    v_data = torch.zeros((qkv_shape//3, qkv_shape//3))
    eye_shape = int(v_proportion * v_data.shape[0])
    v_data[:eye_shape, :eye_shape] = torch.eye(eye_shape)
    model.transformer_encoder.layers[attention_block].self_attn.in_proj_weight.data[2*(qkv_shape//3):] = v_data
    
    # So, (QK^T)V just adds the same vector (first word embedding) to each word in the sequence.  

    # Linear layer at the end of MHA - set to small value to not 'skew' embeddings too much
    model.transformer_encoder.layers[attention_block].self_attn.out_proj.weight.data = sequence_token_weight*torch.eye(qkv_shape//3)
    

def make_imprint_layer(model, measurement, mean, std, block_num=0, self_attn=False):
    '''
    measurement is the Gaussian vector we take inner product w.r.t.
    mean, std = mean, std of features from feature_distribution
    '''
    
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

    bin_dim = model.transformer_encoder.layers[block_num].linear1.weight.data.shape[0]
    bins = _get_bins(mean, std, bin_dim)
    model.transformer_encoder.layers[block_num].linear1.weight.data = measurement.repeat(bin_dim, 1)
    model.transformer_encoder.layers[block_num].linear1.bias.data = _make_biases(model.transformer_encoder.layers[0].linear1.bias, bins)
    
    # We set the second linear layer in the ff to minimally modify (b/c skip connection) 
    model.transformer_encoder.layers[block_num].linear2.weight.data =\
        torch.zeros_like(model.transformer_encoder.layers[block_num].linear2.weight.data)
    model.transformer_encoder.layers[block_num].linear2.weight.data[0] =\
        0.001 * torch.ones_like(model.transformer_encoder.layers[block_num].linear2.weight.data[0])
    model.transformer_encoder.layers[block_num].linear2.bias.data =\
        torch.zeros_like(model.transformer_encoder.layers[block_num].linear2.bias.data)

    if not self_attn:
        # Turn off MHA in this layer for now. We can always turn it back on if it's used
        model.transformer_encoder.layers[block_num].self_attn.out_proj.weight.data =\
            torch.zeros_like(model.transformer_encoder.layers[block_num].self_attn.out_proj.weight.data)

        model.transformer_encoder.layers[block_num].self_attn.out_proj.bias.data =\
            torch.zeros_like(model.transformer_encoder.layers[block_num].self_attn.out_proj.bias.data)

def recover_from_group(model, server, group, no_pos_recs, leaked_tokens):
    indcs = []
    corrs = torch.zeros((len(no_pos_recs), len(group))) 

    # We need to find out what word led to what positionally encoded representation. 
    # Let's try the naive greedy search for correlations between no_pos and with_pos as defined above
    for i, no_p in enumerate(no_pos_recs):
        max_corr = 0
        for j, with_p in enumerate(group):
            val = np.corrcoef(no_p.detach().numpy(), with_p)[0,1]
            corrs[i,j] = val

    # Find which positionally-encoded vector associates with un-positionally-encoded vector
    row_ind, col_ind = linear_sum_assignment(corrs.numpy(), maximize=True)

    order = [(row_i, col_i) for (row_i, col_i) in zip(row_ind, col_ind)]
    order = sorted(order, key=lambda x: x[1])

    # Now let's re-sort the tokens by this order
    sorted_tokens1 = [leaked_tokens[order[i][0]] for i in range(len(order))]

   # Now that we've 'lined-up' the pos-encoded features with non-pos-encoded features, let's subtract the two
    # to get some 'faux' positions (layer norm means they aren't exact).
    estimated_pos = torch.stack([group[order[i][1]] - no_pos_recs[order[i][0]] for i in range(len(order))])
    new_with_pos = [group[order[i][1]] for i in range(len(order))]

    # Now let's get just the additive part of the positional encoding
    dummy_inputs = torch.zeros_like(next(iter(server.external_dataloader))['input_ids'])
    just_pos = torch.stack([model.transformer_encoder.layers[0].norm1(model.pos_encoder(torch.zeros_like(model.encoder(dummy_inputs)) * math.sqrt(model.encoder.embedding_dim)))]).cpu().squeeze()

    just_pos = just_pos.view(-1, just_pos.shape[-1])
    order_coeffs = torch.zeros((len(estimated_pos), len(just_pos)))
    for i in range(len(estimated_pos)):
        for j in range(len(just_pos)):
            order_coeffs[i,j] = np.corrcoef(estimated_pos[i].detach().numpy(), just_pos[j].detach().numpy())[0,1]
    row_ind, col_ind = linear_sum_assignment(order_coeffs.numpy(), maximize=True)
    pos_order = [(row_i, col_i) for (row_i, col_i) in zip(row_ind, col_ind)]
    pos_order = sorted(pos_order, key=lambda x: x[1])
    return torch.stack([sorted_tokens1[pos_order[i][0]] for i in range(len(pos_order))])
    
