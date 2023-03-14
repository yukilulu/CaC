import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from utils import *
import time

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    # print(dsize, tr_size, dsize - tr_size)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(
        dsets["source_tr"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(
        dsets["source_te"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )

    return dset_loaders


def cal_acc(loader, fea_bank, socre_bank, netF, netB, netC, args, flag=False):
    start_test = True
    num_sample = len(loader.dataset)
    nu = []
   

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            indx = data[-1]
            inputs = inputs.cuda()
            fea = netB(netF(inputs))


            outputs = netC(fea)
            softmax_out = nn.Softmax(dim=-1)(outputs)
            nu.append(torch.mean(torch.svd(softmax_out)[1]))
            output_f_norm = F.normalize(fea)
            if start_test:
                all_output = outputs.float().cpu()

                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )

   # _, socre_bank_ = torch.max(socre_bank, 1)
   # distance = fea_bank.cpu() @ fea_bank.cpu().T
   # _, idx_near = torch.topk(distance, dim=-1, largest=True, k=4)
   # score_near = socre_bank_[idx_near[:, :]].float().cpu()  # N x 4

    if True:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = " ".join(aa)
        if True:
            return aacc, acc  # , acc1, acc2#, nu_mean, s10_avg

    else:
        return accuracy * 100, mean_ent


def hyper_decay(x, beta=-2, alpha=1):
    weight = (1 + 10 * x) ** (-beta) * alpha
    return weight


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    modelpath = args.output_dir_src + "/source_F.pt"
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + "/source_B.pt"
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + "/source_C.pt"
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        # if k.find('bn')!=-1:
        if True:
            param_group += [{"params": v, "lr": args.lr * 0.1}]  # 0.1

    for k, v in netB.named_parameters():
        if True:
            param_group += [{"params": v, "lr": args.lr * 1}]  # 1
    for k, v in netC.named_parameters():
        param_group_c += [{"params": v, "lr": args.lr * 1}]  # 1

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    # building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    bank_size = (num_sample * args.memory) // 100
    # bank_size = num_sample
    fea_bank = torch.randn(bank_size, 256).cuda()
    score_bank = torch.randn(bank_size, args.class_num).cuda()

    near_bank = torch.zeros(bank_size, args.K).long().cuda()
    near_bank -= 1
    index_bank = torch.randint(bank_size, size=(bank_size,)).cuda()

    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[-1]
            # labels = data[1]
            inputs = inputs.cuda()
            bs = inputs.size(0)
            output = netB(netF(inputs))
            output_norm = F.normalize(output)
            outputs = netC(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank = torch.cat((fea_bank[bs:, :], output_norm.detach().clone()), dim=0)
            score_bank = torch.cat((score_bank[bs:, :], outputs.detach().clone()), dim=0)  # .cpu()
            index_bank = torch.cat((index_bank[bs:], indx.cuda()), dim=0)  # .cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    #old_bank = score_bank

    netF.train()
    netB.train()
    netC.train()
    acc_log = 0

    real_max_iter = max_iter

    while iter_num < real_max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_target = inputs_test.cuda()
        if True:
            alpha = (max_iter / (max_iter+ iter_num)) ** (args.beta) * args.alpha
        else:
            alpha = args.alpha

        iter_num += 1
        if args.lr_decay:
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
            lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)
     
        features_test = netB(netF(inputs_target))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        bs = softmax_out.size(0)
        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            fea_bank = torch.cat((fea_bank[bs:, :], output_f_norm), dim=0)
            score_bank = torch.cat((score_bank[bs:, :], softmax_out), dim=0)  # .cpu()
            index_bank = torch.cat((index_bank[bs:], tar_idx.cuda()), dim=0)  # .cpu()
            distance = output_f_norm @ fea_bank.T
            _, indices = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
            idx_near = index_bank[indices]

            #The nearest neighbors of the current sample are stored in the memory bank
            near_bank = torch.cat((near_bank[bs:, :], idx_near[:, 1:].cuda()), dim=0)
           
            idx_double = near_bank[indices[:, 1:]]  # bs,k,k
            tar_idx_ = tar_idx.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # The nearest neighbors of sample x in the current batch have matching conditions
            match = (idx_double.unsqueeze(-1) == tar_idx_.cuda()).long()

            #‘idx_match_in_cur’ is the index corresponding to this x
            idx_match_in_cur = match[0]

            # ‘idx_match_project_cur’: The nearest neighbor of x is equal to the index corresponding to the tar_idx
            idx_match_project_cur = match[-1]
            # first_k = match[1]
            weight = torch.ones(bs, bs).cuda()
            weight[idx_match_in_cur, idx_match_project_cur] = 0.0

        dot = torch.matmul(score_bank[indices[:,1:]],softmax_out.transpose(0,1)).transpose(0,1) #[k,b,b]
        near = torch.diagonal(dot, dim1=1, dim2=2).sum(0)  # ([k,b])->([5, 64])
        mask = torch.ones((args.K, inputs_test.shape[0], inputs_test.shape[0])).cuda()
        diag_num = torch.diagonal(mask, dim1=1, dim2=2)
        mask_diag = torch.diag_embed(diag_num, dim1=1, dim2=2)
        mask = mask - mask_diag
        # Whether to remove similar samples in the negative sample pool
        if args.weight:
            neg = torch.sum((dot * mask).mean(dim=0) * weight, dim=1)
        else:
            neg = torch.sum((dot * mask).mean(dim=0), dim=1)
        cac_loss = -near + neg * alpha
        loss = torch.mean(cac_loss)

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()
        
        if iter_num % interval_iter == 0 or iter_num == max_iter:

            netF.eval()
            netB.eval()
            netC.eval()
            if args.dataset == "visda":
                acc, accc = cal_acc(
                    dset_loaders["test"],
                    fea_bank,
                    score_bank,
                    netF,
                    netB,
                    netC,
                    args,
                    flag=True,
                )
                log_str = (
                        "Task: {}, Iter:{}/{};  Acc on target: {:.2f}".format(
                            args.name, iter_num, max_iter, acc
                        )
                        + "\n"
                        + "T: "
                        + accc
                )

            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")
            netF.train()
            netB.train()
            netC.train()
            """if acc>acc_log:
                acc_log = acc
                torch.save(
                    netF.state_dict(),
                    osp.join(args.output_dir, "target_F_" + '2021_'+str(args.tag) + ".pt"))
                torch.save(
                    netB.state_dict(),
                    osp.join(args.output_dir,
                                "target_B_" + '2021_' + str(args.tag) + ".pt"))
                torch.save(
                    netC.state_dict(),
                    osp.join(args.output_dir,
                                "target_C_" + '2021_' + str(args.tag) + ".pt"))"""

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPA")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=15, help="max iterations")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument("--dataset", type=str, default="visda")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lr_decay", default=True, action="store_true")
    parser.add_argument("--net", type=str, default="resnet101")
    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--output", type=str, default="weight/target/")
    parser.add_argument("--output_src", type=str, default="weight/source/")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=18.0)
    parser.add_argument("--alpha_decay", default=True)
    parser.add_argument("--memory", type=int, default=100)
    parser.add_argument("--weight", default=False,action="store_true")
    args = parser.parse_args()

    if args.dataset == "office-home":
        names = ["Art", "Clipart", "Product", "RealWorld"]
        args.class_num = 65
    if args.dataset == "visda":
        names = ["train", "validation"]
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = '/export/home/liyg/code/SFDA/'
        args.s_dset_path = folder + args.dataset + "/" + names[args.s] + "_list.txt"
        args.t_dset_path = folder + args.dataset + "/" + names[args.t] + "_list.txt"
        args.test_dset_path = folder + args.dataset + "/" + names[args.t] + "_list.txt"

        args.output_dir_src = osp.join(
            args.output_src, args.dataset, names[args.s][0].upper()
        )
        args.output_dir = osp.join(
            args.output
        )
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system("mkdir -p " + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(osp.join(args.output_dir, "-cac-weight"+ str(args.weight) +"_K" + str(args.K) + "_beta" + str(args.beta) +"_memory" + str(args.memory) + ".txt"), "w")
        args.out_file.write(print_args(args) + "\n")
        args.out_file.flush()
        train_target(args)

