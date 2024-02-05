import numpy as np
import torch
import torch.nn as nn
import os


def EntropyLoss(input_, attention_global):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    attention_global_out = torch.masked_select(attention_global, mask)
    entropy = -(torch.sum(attention_global_out.data * mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def LAN(input_list, ad_net_list, grl_layer_list, ad_net_global, grl_layer_global, max_iter, iter_num, len_train_source,
        use_gpu=True, ad_local_num=7, conv_feature_dim=2048, network="ResNet"):
    base_network = input_list[1]
    batch_size = input_list[0].size(0) // 2
    dc_target = torch.Tensor(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())  # domain label
    if use_gpu:
        dc_target = dc_target.cuda()
    conv_feature_list = input_list[0]
    loss = 0
    loss_conv = []

    local_weight = 0.0
    for i in range(len(ad_net_list)):
        outer_product = conv_feature_list.narrow(2, i, 1).squeeze(2)
        ad_out = ad_net_list[i](grl_layer_list[i].apply(outer_product, max_iter, iter_num))
        iter_loss_local = nn.BCELoss()(ad_out.float().view(-1), dc_target.float().view(-1))
        loss += iter_loss_local
        Adistance_local = 2 * (1 - 1 * iter_loss_local)
        local_weight += Adistance_local
        ad_entropy_local = - ad_out * torch.log2(ad_out + 1e-10) - (1 - ad_out) * torch.log2(1 - ad_out + 1e-10)  # v1
        ad_entropy_local = 1 - ad_entropy_local
        loss_conv.append(ad_entropy_local)

    attention_region = torch.cat(loss_conv, 1)
    attention_region = 1 + attention_region.view(-1, 1, ad_local_num * ad_local_num)
    local_weight /= len(ad_net_list)

    attention_pool = conv_feature_list * attention_region.data
    attention_pool = attention_pool.view(-1, conv_feature_dim, ad_local_num, ad_local_num)
    attention_pool_local = attention_pool.view(len(attention_pool), -1)

    if "ResNet" in network:
        fc_features = base_network.bottleneck(attention_pool_local)
    elif "AlexNet" in network:
        fc_features = base_network.classifier(attention_pool_local)
        if base_network.use_bottleneck:
            fc_features = base_network.bottleneck(fc_features)

    transfer_loss_region = loss / float(ad_local_num * ad_local_num)
    ad_out_global = ad_net_global(grl_layer_global.apply(fc_features, max_iter, iter_num))
    attention_global = - ad_out_global * torch.log2(ad_out_global + 1e-10) - (1 - ad_out_global) * torch.log2(
        1 - ad_out_global + 1e-10)
    attention_global = 1 + attention_global
    transfer_loss_global = nn.BCELoss()(ad_out_global.float().view(-1), dc_target.float().view(-1))

    global_weight = 2 * (1 - 1 * transfer_loss_global)

    if iter_num < len_train_source:
        mu = 0.5
    else:
        mu = local_weight / (global_weight + local_weight)
        if mu <= 0 or mu >= 1:
            mu = 0.5
    transfer_loss = (1 - mu) * transfer_loss_global + mu * transfer_loss_region
    return transfer_loss, base_network.fc(fc_features), attention_global
