# coding: utf-8
import argparse
import os
import os.path as osp
import shutil
import time

import loss
import lr_schedule
import matplotlib.pyplot as plt
import network
import numpy as np
np.random.seed(2018)
import pre_process as prep
import torch
torch.manual_seed(2018)

torch.cuda.manual_seed_all(2018)
os.environ['PYTHONHASHSEED'] = str(2018)
##
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as util_data
from data_list import ImageList
from tensorboardX import SummaryWriter
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

plt.switch_backend('agg')
from utils import str2bool
from sklearn.metrics import confusion_matrix

optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam}

def kappa(confusion_matrix):
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)

def test_wrapper(loader, model, ad_net_list, test_10crop=True, gpu=True, ad_local_num=7, conv_feature_dim=2048):
    start_test = True
    if test_10crop:
        #print("----------------------test_10crop---------------------")
        iter_test = [iter(loader['test' + str(i)]) for i in range(1)]#iter(loader['test0'])
        for i in range(len(loader['test0'])):#300
            data = [iter_test[j].next() for j in range(1)]
            inputs = [data[j][0] for j in range(1)]
            labels = data[0][1]
            if gpu:
                for j in range(1):
                    inputs[j] = torch.as_tensor(inputs[j].cuda())
                labels = torch.as_tensor(labels.cuda())
            else:
                for j in range(1):
                    inputs[j] = torch.as_tensor(inputs[j])
                labels = torch.as_tensor(labels)
            outputs = []
            for j in range(1):
                _conv = model(inputs[j])
                conv_feature_list = _conv.view(_conv.shape[0], _conv.shape[1], ad_local_num * ad_local_num)
                attention_region = []
                for k in range(len(ad_net_list)):
                    outer_product = conv_feature_list.narrow(2, k, 1).squeeze(2)
                    ad_net_list[k].eval()
                    ad_out = ad_net_list[k](outer_product)
                    ad_entropy_local = - ad_out * torch.log2(ad_out + 1e-10) - (1 - ad_out) * torch.log2(1 - ad_out + 1e-10)
                    ad_entropy_local = 1 - ad_entropy_local
                    attention_region.append(ad_entropy_local)
                attention_region = torch.cat(attention_region, 1)
                attention_region = 1 + attention_region.view(-1, 1, ad_local_num * ad_local_num)
                attention_pool = conv_feature_list * attention_region.data
                attention_pool = attention_pool.view(-1, conv_feature_dim, ad_local_num, ad_local_num)
                attention_pool_local = attention_pool.view(len(attention_pool), -1)
                fc_features = model.bottleneck(attention_pool_local)
                predict_out = model.fc(fc_features)
                outputs.append(nn.Softmax(dim=1)(predict_out))
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        raise ValueError("必须使用test_10crop")

    output_np = all_output.cpu().numpy()
    label_source_np = all_label.cpu().numpy()
    predict_source_np = np.argmax(output_np, axis=1)

    C_M = confusion_matrix(label_source_np, predict_source_np)
    Kappa = kappa(C_M)

    sum_2foot = 0
    per_class = []
    for I in range(C_M.shape[0]):
        sum_2foot += C_M[I][I]
        classI = C_M[I][I] / sum(C_M[I])
        per_class.append(classI)
    AA = sum_2foot/label_source_np.shape[0]
    OA = sum(per_class)/len(per_class)

    return AA, label_source_np, predict_source_np, Kappa,OA,per_class


def train_wrapper(config):
    tensor_writer = SummaryWriter(config["tensorboard_path"])

    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train( resize_size=prep_config["resize_size"], crop_size=prep_config["crop_size"])
    prep_dict["target"] = prep.image_train( resize_size=prep_config["resize_size"], crop_size=prep_config["crop_size"])
    prep_dict["source_test"] = prep.image_test( resize_size=prep_config["resize_size"], crop_size=prep_config["crop_size"])
    prep_dict["target_test"] = prep.image_test( resize_size=prep_config["resize_size"], crop_size=prep_config["crop_size"])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop( resize_size=prep_config["resize_size"], crop_size=prep_config["crop_size"])
    else:
        prep_dict["test"] = prep.image_test( resize_size=prep_config["resize_size"], crop_size=prep_config["crop_size"])

    class_criterion = nn.CrossEntropyLoss()
    entropy_criterion = loss.EntropyLoss
    loss_params = config["loss"]

    datadir = config['datadir']
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    print("The number of images in source domain is: ")
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), transform=prep_dict["source"], datadir=datadir)
    dset_loaders["source"] = util_data.DataLoader(dsets["source"],batch_size=data_config["source"]["batch_size"],shuffle=True, num_workers=6)

    print("The number of images in target domain is: ")
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(),transform=prep_dict["target"], datadir=datadir)
    dset_loaders["target"] = util_data.DataLoader(dsets["target"],batch_size=data_config["target"]["batch_size"], shuffle=True, num_workers=6)

    if prep_config["test_10crop"]:
        for i in range(1):
            dsets["test" + str(i)] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                               transform=prep_dict["test"]["val" + str(i)], datadir=datadir)
            dset_loaders["test" + str(i)] = util_data.DataLoader(dsets["test" + str(i)], \
                                                                 batch_size=data_config["test"]["batch_size"], \
                                                                 shuffle=False, num_workers=6)
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"], datadir=datadir)
        dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                                    batch_size=data_config["test"]["batch_size"], \
                                                    shuffle=False, num_workers=6)
        dsets["target_test"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                         transform=prep_dict["test"], datadir=datadir)
        dset_loaders["target_test"] = util_data.DataLoader(dsets["target_test"], \
                                                           batch_size=data_config["test"]["batch_size"], \
                                                           shuffle=False, num_workers=6)

    print("data preparation is finished")


    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()


    if net_config["params"]["new_cls"]:
        if net_config["params"]["use_bottleneck"]:
            parameter_list = [{"params": base_network.feature_layers.parameters(), "lr": 1}, \
                              {"params": base_network.bottleneck.parameters(), "lr": 10}, \
                              {"params": base_network.fc.parameters(), "lr": 10}]
        else:
            parameter_list = [{"params": base_network.feature_layers.parameters(), "lr": 1}, \
                              {"params": base_network.fc.parameters(), "lr": 10}]
    else:
        parameter_list = [{"params": base_network.parameters(), "lr": 1}]
    print("parameters are collected")


    ad_net_list = []
    gradient_reverse_layer_list = []
    ad_local_num = config["ad_local_num"]
    for i in range(ad_local_num * ad_local_num):
        if config["dataset"] == "caltech" or config["dataset"] == "caltech-office":
            ad_net = network.SmallAdversarialNetwork(base_network.output_num())
        elif config["dataset"] == "imagenet":
            ad_net = network.LittleAdversarialNetwork(base_network.output_num())
        else:
            ad_net = network.AdversarialNetwork(base_network.output_num())
        gradient_reverse_layer = network.AdversarialLayer()
        gradient_reverse_layer_list.append(gradient_reverse_layer)
        if use_gpu:
            ad_net = ad_net.cuda()
        ad_net_list.append(ad_net)
        parameter_list.append({"params": ad_net.parameters(), "lr": 10})
    print("AdversarialNetwork is set")

    ad_net_global = network.AdversarialNetwork(config['bottleneck_dim'])
    gradient_reverse_layer_global = network.AdversarialLayer()
    if use_gpu:
        ad_net_global = ad_net_global.cuda()
    parameter_list.append({"params": ad_net_global.parameters(), "lr": 10})

    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    print("optimizer is set")

    len_train_source = len(dset_loaders["source"]) - 1
    len_train_target = len(dset_loaders["target"]) - 1
    best_acc = 0.0
    iter_num = 0.0
    for i in range(config["num_iterations"]):

        start_time = time.time()
        iter_num += 1.0
        if i % config["test_interval"] == 0 and i > 0:
            base_network.train(False)

            valid_acc, label_source_np, predict_source_np, Kappa, OA, per_class = test_wrapper(dset_loaders,
                                                                         base_network,
                                                                         ad_net_list,
                                                                         test_10crop=prep_config["test_10crop"],
                                                                         gpu=use_gpu,
                                                                         ad_local_num=config["ad_local_num"],
                                                                         conv_feature_dim=config["conv_feature_dim"])
            tensor_writer.add_scalar('AA', valid_acc.item(), i)
            tensor_writer.add_scalar('Kappa', Kappa.item(), i)
            tensor_writer.add_scalar('OA', OA.item(), i)
            tensor_writer.add_scalar('per_class', sum(per_class), i)

            tensor_writer.file_writer.flush()

            temp_model = nn.Sequential(base_network)
            if valid_acc >= best_acc:
                best_acc = valid_acc
                best_model = temp_model
                torch.save(best_model.state_dict(), osp.join(config["output_path"], "best_model.pth.tar"))
                print("best model is saved", i)
                best_iter=i
            log_str = "iter: {:04d}, AA: {:.5f}, best_acc: {:.5f} (inter {:04d}), Kappa: {:.5f}, OA: {:.5f}, per_class: {}\n".format(i, valid_acc, best_acc, best_iter, Kappa, OA, per_class)
            config["out_file"].write(log_str)
            config["out_file"].flush()
            print(log_str)

        base_network.train(True)

        for param in base_network.feature_layers.parameters():
            param.requires_grad = config['feature_finetune']
        for param in base_network.bottleneck.parameters():
            param.requires_grad = True
        for param in base_network.fc.parameters():
            param.requires_grad = True

        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source, _ = iter_source.next()
        inputs_target, labels_target, _ = iter_target.next()
        if use_gpu:
            inputs_source, inputs_target, labels_source = \
                torch.as_tensor(inputs_source).cuda(), torch.as_tensor(inputs_target).cuda(), \
                torch.as_tensor(labels_source).cuda()
        else:
            print("use cpu")
            inputs_source, inputs_target, labels_source = torch.as_tensor(inputs_source), \
                                                          torch.as_tensor(inputs_target), torch.as_tensor(labels_source)

        for ad_net in ad_net_list:
            ad_net.train(True)

        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        conv_x = base_network(inputs)
        conv_feature_list = conv_x.view(conv_x.shape[0], conv_x.shape[1], ad_local_num * ad_local_num)  ##72*2048*49

        transfer_loss, outputs, attention_global = loss.LAN([conv_feature_list, base_network],
                                                            ad_net_list,gradient_reverse_layer_list,
                                                            ad_net_global,gradient_reverse_layer_global,
                                                            config['max_iter'],iter_num,len_train_source,
                                                            use_gpu, ad_local_num,
                                                            config["conv_feature_dim"],config["network_name"])

        entropy_loss = entropy_criterion(nn.Softmax(dim=1)(outputs), attention_global)

        classifier_loss = class_criterion(outputs.narrow(0, 0, int(outputs.size(0) / 2)), labels_source)
        total_loss = classifier_loss + transfer_loss + loss_params["entropy_trade_off"] * entropy_loss

        output_source = outputs.narrow(0, 0, int(outputs.size(0) / 2))

        predict_source_np = output_source.cpu().data.numpy()
        labels_source_np = labels_source.cpu().data.numpy()
        predict_source_np = np.argmax(predict_source_np, axis=1)
        correct_pred = np.equal(predict_source_np, labels_source_np)
        train_acc = np.mean(correct_pred.astype(float))

        total_loss.backward()
        optimizer.step()
        tensor_writer.add_scalar('total_loss', total_loss, i)
        tensor_writer.add_scalar('transfer_loss', transfer_loss, i)
        tensor_writer.add_scalar('classifier_loss', classifier_loss, i)
        tensor_writer.add_scalar('entropy_loss', entropy_loss, i)
        tensor_writer.add_scalar('train_acc', train_acc.item(), i)

        tensor_writer.file_writer.flush()
        if not isinstance(transfer_loss, int) and not isinstance(transfer_loss, float):
            transfer_loss = transfer_loss.data.item()
        if not isinstance(classifier_loss, int) or not isinstance(classifier_loss, float):
            classifier_loss = classifier_loss.data.item()
        if not isinstance(entropy_loss, int) or not isinstance(entropy_loss, float):
            entropy_loss = entropy_loss.data.item()
        if i % 10 == 0 and i > 0:
            print("inter{:3d}|train_acc:{:.4f} |classifier_loss:{:.4f} |transfer_loss:{:.4f} |entropy_loss:{:.4f} |runing_time:{:.4f}"
                  .format(i, train_acc.item(), classifier_loss, transfer_loss, entropy_loss, time.time() - start_time))
    return best_acc


if __name__ == "__main__":
    config = {}
    config["num_iterations"] = 20000
    parser = argparse.ArgumentParser(description='Partial Transfer Learning with Selective Adversarial Networks')
###
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--Output_str', type=str, default='MRDAN_A2U', help="entropy loss trade off")

    parser.add_argument('--entropy_trade_off', type=float, default=0.1, help="entropy loss trade off")
    parser.add_argument('--test_interval', type=int, default=200, help="interval of two continuous test phase")
    parser.add_argument('--max_iter', type=int, default=1500)
###
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--datadir', type=str, default='/root/autodl-tmp/dataMRDAN/', help="the root datadir of the dataset")
    parser.add_argument('--dset', type=str, default='cro-scene', help="The dataset or source dataset used")
    
    parser.add_argument('--source_dset_path', type=str, default='/root/autodl-tmp/dataMRDAN/AID.txt',
                        help="The source dataset path list")
    parser.add_argument('--target_dset_path', type=str, default='/root/autodl-tmp/dataMRDAN/UCM.txt',
                        help="The target dataset path list")

    parser.add_argument('--batch_size', type=int, default=36)#36
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    ##
    parser.add_argument('--lr_gama', type=float, default=0.001, help="gama variable for selective transfer loss")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")

    ##
    parser.add_argument('--momentum', type=float, default=0.95, help="momentum for SGD")
    parser.add_argument('--optim_type', type=str, default='SGD', help="the type of optimizer")
    parser.add_argument('--feature_finetune', type=str2bool, default=True, help="feature extractor finetune or not")
    parser.add_argument('--debug_str', type=str, default='')
    parser.add_argument('--loss_type', type=str, default='LAN', help="loss type")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    config["test_interval"] = args.test_interval
    config["loss_type"] = args.loss_type
    config['datadir'] = args.datadir
    config['optim_type'] = args.optim_type
    config['feature_finetune'] = args.feature_finetune
    config['max_iter'] = args.max_iter
    config['bottleneck_dim'] = args.bottleneck_dim
    output_str = args.Output_str

    debug_str = args.debug_str
    batch_size = args.batch_size
    bottleneck_dim = args.bottleneck_dim
    lr_gama = args.lr_gama
    source_flag, target_flag = osp.basename(args.source_dset_path)[0].upper(), osp.basename(args.target_dset_path)[
        0].upper()

    if (debug_str != "" and debug_str != None):
        output_str += ",debug-%s" % (debug_str)
    config["output_path"] = "./checkpoint/" + output_str
    config["results_path"] = "./results/" + output_str
    config["tensorboard_path"] = "./vis2w/" + output_str

    if osp.exists(config["output_path"]) or osp.exists(config["results_path"]) or osp.exists(
            config["tensorboard_path"]):
        a ='y'
        if a.lower() == "y":
            pass
        else:
            exit()
    if osp.exists(config["output_path"]):
        shutil.rmtree(config["output_path"])
    os.makedirs(config["output_path"])
    if osp.exists(config["results_path"]):
        shutil.rmtree(config["results_path"])
    os.makedirs(config["results_path"])
    if osp.exists(config["tensorboard_path"]):
        shutil.rmtree(config["tensorboard_path"])
    os.makedirs(config["tensorboard_path"])

    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    ## configurations for pre_process,including size of the image and crop size
    config["prep"] = {"test_10crop": True, "resize_size": 224, "crop_size": 224}

    ## configurations for optimizer
    if config['optim_type'] == 'SGD':
        config["optimizer"] = {"type": "SGD", "optim_params": {"lr": 1.0, "momentum": args.momentum, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                               "lr_type": "inv", \
                               "lr_param": {"init_lr": args.lr, "gamma": lr_gama, "power": 0.75}}
    elif config['optim_type'] == 'Adam':
        config["optimizer"] = {"type": "Adam", "optim_params": {"lr": 1.0}, \
                               "lr_type": "inv", \
                               "lr_param": {"init_lr": args.lr, "gamma": lr_gama, "power": 0.75}}

    config["loss"] = {"entropy_trade_off": args.entropy_trade_off}      #0.1

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.source_dset_path, "batch_size": batch_size}, \
                      "target": {"list_path": args.target_dset_path, "batch_size": batch_size}, \
                      "test": {"list_path": args.target_dset_path, "batch_size": 8}}#target_test_path

    config["dataset"] == "cro-scene"
    config["class_num"] = 6
    config["network_name"] = args.net

    if "ResNet" in args.net:
        config["conv_feature_dim"] = 2048
        config["ad_local_num"] = 7
        config["network"] = {"name": network.ResNetFc, "params": {"resnet_name": args.net,
                                                                  "use_bottleneck": True,
                                                                  "bottleneck_dim": bottleneck_dim, # 256
                                                                  "new_cls": True,
                                                                  "class_num": config["class_num"],
                                                                  "conv_feature_dim": config["conv_feature_dim"], #2048
                                                                  "ad_local_num": config["ad_local_num"]}}  #7

    elif "AlexNet" in args.net:
        config["conv_feature_dim"] = 256
        config["ad_local_num"] = 6
        config["network"] = {"name": network.AlexNetFc, "params": {"use_bottleneck": True,
                                                                   "bottleneck_dim": bottleneck_dim,
                                                                   "new_cls": True,
                                                                   "class_num": config["class_num"],
                                                                   "conv_feature_dim": config["conv_feature_dim"],
                                                                   "ad_local_num": config["ad_local_num"]}}
    print(config["class_num"])

    train_wrapper(config)
