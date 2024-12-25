import gc
import csv
import argparse
import statistics
from tqdm import tqdm
import numpy as np
from torch import nn
import torch.utils.data
from torchvision.datasets import CIFAR10

from Models.RED_CNN_add import sc_encoder, sc_decoder, Quantization
from Models.SD_model_add import DenseEncoder, DenseDecoder, SRNet, ReDecoder, accuracy
from Models.utils_s import data_tr_1, sr_Net_data, get_payload

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def net_to_device(device):
    SC_encoder.to(device)
    SC_decoder.to(device)
    critic1.to(device)
    critic2.to(device)
    Ser_net.to(device)
    Sdr_net.to(device)
    re_decoder.to(device)


def _transmit(im, payload, is_train=False, shuffle=False):
    imc = SC_encoder(im)
    ims = Ser_net(imc.detach(), payload.detach())
    if shuffle:
        idx = torch.randperm(ims.shape[0])
        ims = ims[idx, :]
    imc_max = torch.round(torch.max(torch.abs(imc)))
    imc = imc / imc_max
    ims = ims / imc_max
    if is_train:
        imc = Quantization()(imc)
        ims = Quantization()(ims)
    else:
        imc = torch.clamp(imc, -1, 1)
        ims = torch.clamp(ims, -1, 1)
        ims = (255.0 * (ims + 1.0) / 2.0).type(torch.uint8)
        imc = (255.0 * (imc + 1.0) / 2.0).type(torch.uint8)
        ims = 2.0 * ims.float() / 255.0 - 1.0
        imc = 2.0 * imc.float() / 255.0 - 1.0
    imc = imc * imc_max
    ims = ims * imc_max
    return imc, ims


def _receiver(imc, ims, is_train=False):
    out_data = Sdr_net(ims)

    im_s = SC_decoder(ims)
    im_c = SC_decoder(imc)

    if is_train:
        im_c = Quantization()(im_c)
        im_s = Quantization()(im_s)
    else:
        im_c = torch.clamp(im_c, -1, 1)
        im_s = torch.clamp(im_s, -1, 1)
        im_s = (255.0 * (im_s + 1.0) / 2.0).type(torch.uint8)
        im_c = (255.0 * (im_c + 1.0) / 2.0).type(torch.uint8)
        im_s = 2.0 * im_s.float() / 255.0 - 1.0
        im_c = 2.0 * im_c.float() / 255.0 - 1.0

    out_data2 = re_decoder(im_s)
    return im_c, im_s, out_data, out_data2


def _val(validate, metrics, device):
    gc.collect()
    Ser_net.eval()
    Sdr_net.eval()
    SC_encoder.eval()
    SC_decoder.eval()
    critic1.eval()
    critic2.eval()
    re_decoder.eval()
    pbar = tqdm(validate)
    all_labels1 = []
    all_labels2 = []
    all_outputs1 = []
    all_outputs2 = []
    for im, _ in pbar:
        with torch.no_grad():
            im = im.to(device)
            if bpp < 1:
                payload01 = torch.zeros((im.shape[0], 1, 1, round(bpp * H * W))).random_(0, 2).to(device)
                payload = get_payload(payload01, bpp, H, W, device)
            else:
                payload = torch.zeros((im.shape[0], int(bpp), H, W)).random_(0, 2).to(device)
            imc, ims = _transmit(im, payload, False,False)
            ##### channel #######
            im_c, im_s, out_data, out_data2 = _receiver(imc, ims, False)

            # score
            imc_max = torch.round(torch.max(torch.abs(imc)))
            im_train, label_train1 = sr_Net_data((imc/imc_max).float(), (ims/imc_max).float())
            im_train = im_train.to(device)
            label_train1 = label_train1.to(device)
            label_out1 = critic1(im_train)
            acc1 = accuracy(label_out1, label_train1)

            im_train, label_train2 = sr_Net_data(im_c.float(), im_s.float())
            im_train = im_train.to(device)
            label_train2 = label_train2.to(device)
            label_out2 = critic2(im_train)
            acc2 = accuracy(label_out2, label_train2)

            metrics['v.acc1'].append(acc1.item())
            metrics['v.acc2'].append(acc2.item())

            all_outputs1.extend(label_out1.detach().cpu().numpy())
            all_labels1.extend(label_train1.detach().cpu().numpy())
            all_outputs2.extend(label_out2.detach().cpu().numpy())
            all_labels2.extend(label_train2.detach().cpu().numpy())
            torch.cuda.empty_cache()
    #
    all_outputs1 = np.array(all_outputs1)
    all_labels1 = np.array(all_labels1)
    all_outputs2 = np.array(all_outputs2)
    all_labels2 = np.array(all_labels2)

    # ROC AUC
    fpr1, tpr1, _ = roc_curve(all_labels1, all_outputs1[:, 1])
    roc_auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, _ = roc_curve(all_labels2, all_outputs2[:, 1])
    roc_auc2 = auc(fpr2, tpr2)

    # Plot ROC curve
    plt.figure(figsize=(3.2, 3))
    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'Revealing Stage I \n(AUC = {roc_auc1:0.2f})')
    plt.plot(fpr2, tpr2, color='blue', lw=2, label=f'Revealing Stage II \n(AUC = {roc_auc2:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC Curve of Sr-Net')
    plt.legend(loc="lower right", fontsize=9)#
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)
    plt.savefig('./results/roc_curve.eps')
    plt.close()

METRIC_FIELDS = [
    'v.acc1',
    'v.acc2',
]


def train(valid_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net_to_device(device)
    metrics = {field: list() for field in METRIC_FIELDS}
    with open(f'./results/Sema_add2_SRnet{bpp}.csv', 'a', newline='', encoding='utf-8') as f:
        f.truncate(0)
    print(l * 100+num)
    # val
    _val(valid_data, metrics, device)
    fit_metrics = {f'{k}_mean': sum(v) / len(v) for k, v in metrics.items()}
    fit_metrics.update({f'{k}_std': statistics.stdev(v) for k, v in metrics.items() if len(v) > 1})
    with open(f'./results/Sema_add2_SRnet{bpp}.csv', 'a', newline='', encoding='utf-8') as f:
        fieldnames = [f'{field}_mean' for field in METRIC_FIELDS] + [f'{field}_std' for field in METRIC_FIELDS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows([fit_metrics])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpp', type=float, default=1, help='bpp')
    parser.add_argument('--kernel_size1', type=int, default=3, help='kernel_size1')
    parser.add_argument('--kernel_size2', type=int, default=3, help='kernel_size2')
    parser.add_argument('--num_epochs', type=int, default=200, help='num_epochs')
    parser.add_argument('--best_epoch', type=int, default=50, help='best_epoch')
    args = parser.parse_args()

    # argument
    l = 20
    num = 0
    H = 64
    W = 64
    hidden_size = 16
    out_ch = 32
    best_epoch = args.best_epoch
    bpp = args.bpp
    k1 = args.kernel_size1
    k2 = args.kernel_size2
    num_epochs = args.num_epochs
    #### NET ####
    # SC
    SC_encoder = sc_encoder(k1, k2, out_ch)
    SC_decoder = sc_decoder(k1, k2, out_ch)
    # ste
    Ser_net = DenseEncoder(bpp, k1, k2, hidden_size, out_ch)
    Sdr_net = DenseDecoder(bpp, k1, k2, hidden_size, out_ch)
    # re-decode
    re_decoder = ReDecoder(bpp, hidden_size)

    critic1 = SRNet()
    critic2 = SRNet()

    SC_encoder = nn.DataParallel(SC_encoder)
    SC_decoder = nn.DataParallel(SC_decoder)
    Ser_net = nn.DataParallel(Ser_net)
    Sdr_net = nn.DataParallel(Sdr_net)
    re_decoder = nn.DataParallel(re_decoder)

    checkpoint = torch.load(
        f'./model/SD_add2{bpp}{k1}{k2}_%s.pth' % (str(best_epoch)))
    SC_encoder.load_state_dict(checkpoint['SC_encoder'])
    SC_decoder.load_state_dict(checkpoint['SC_decoder'])
    Ser_net.load_state_dict(checkpoint['Ser_net'])
    Sdr_net.load_state_dict(checkpoint['Sdr_net'])
    re_decoder.load_state_dict(checkpoint['re_decoder'])

    checkpoint2 = torch.load(
        f'./model/SrNetA{bpp}{k1}{k2}_%s.pth' % (str(149)))  # load
    critic1.load_state_dict(checkpoint2['critic1'])
    critic2.load_state_dict(checkpoint2['critic2'])

    critic_optimizer = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    critic_optimizer2 = torch.optim.Adam(critic2.parameters(), lr=1e-3)

    # load
    train_set = CIFAR10('./data', train=True, transform=data_tr_1, download=True)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=500, shuffle=True)
    test_set = CIFAR10('./data', train=False, transform=data_tr_1, download=True)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    train(test_data)