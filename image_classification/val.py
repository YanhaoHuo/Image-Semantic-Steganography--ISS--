import gc
import csv
import argparse
from tqdm import tqdm
from torch import nn
import torch.utils.data
from torchvision.datasets import CIFAR10
from torch.nn.functional import mse_loss

from Models.RED_CNN_add import sc_encoder, sc_decoder, Quantization
from Models.GoogleNet import googlenet
from Models.SD_model_add import DenseEncoder, DenseDecoder, ReDecoder
from Models.utils_s import _ssim, get_acc, data_tr_1, get_payload, get_payload01


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


def net_to_device(device):
    SC_encoder.to(device)
    SC_decoder.to(device)
    classifier.to(device)
    Ser_net.to(device)
    Sdr_net.to(device)
    re_decoder.to(device)


def _val(validate, metrics, device):
    gc.collect()
    Ser_net.eval()
    Sdr_net.eval()
    SC_encoder.eval()
    SC_decoder.eval()
    classifier.eval()
    re_decoder.eval()
    pbar = tqdm(validate)
    for im, label in pbar:
        with torch.no_grad():
            im = im.to(device)
            label = label.to(device)
            if bpp < 1:
                payload01 = torch.zeros((im.shape[0], 1, 1, round(bpp * H * W))).random_(0, 2).to(device)
                payload = get_payload(payload01, bpp,H,W, device)
            else:
                payload = torch.zeros((im.shape[0], int(bpp), H, W)).random_(0, 2).to(device)
            imc, ims = _transmit(im, payload, False, False)
            ##### channel #######
            im_c, im_s, out_data, out_data2 = _receiver(imc, ims, False)

            # score
            if bpp < 1:
                decoder_acc = (get_payload01(out_data,bpp,H,W, device) >= 0.0).eq(
                    payload01 >= 0.5).sum().float() / payload01.numel()
                decoder_acc2 = (get_payload01(out_data2,bpp,H,W, device) >= 0.0).eq(
                    payload01 >= 0.5).sum().float() / payload01.numel()
            else:
                decoder_acc = (out_data >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
                decoder_acc2 = (out_data2 >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
            acc_im = get_acc(classifier(im), label)
            acc_s = get_acc(classifier(im_s), label)

        imc_max = torch.round(torch.max(torch.abs(imc)))
        pbar.set_description(
            f"Loss1: {decoder_acc.item():.3f}; Loss4: {decoder_acc2.item():.3f} "
        )
        metrics['v.de_acc'].append(decoder_acc.item())
        metrics['v.de_acc2'].append(decoder_acc2.item())

        metrics['v.psnr1'].append(10 * torch.log10(4 / mse_loss(imc / imc_max, ims / imc_max)).item())
        metrics['v.ssim1'].append(_ssim(imc/imc_max, ims/imc_max, 2).item())
        metrics['v.psnr2'].append(10 * torch.log10(4 / mse_loss(im_c, im_s)).item())
        metrics['v.ssim2'].append(_ssim(im_c, im_s, 2).item())

        metrics['v.psnr_s'].append(10 * torch.log10(4 / mse_loss(im_s, im)).item())
        metrics['v.ssim_s'].append(_ssim(im_s, im, 2).item())
        metrics['v.class_acc_gap'].append(acc_im-acc_s)
        metrics['v.class_acc_im'].append(acc_im)


METRIC_FIELDS = [
    'v.de_acc',
    'v.de_acc2',

    'v.psnr1',
    'v.ssim1',
    'v.psnr2',
    'v.ssim2',

    'v.psnr_s',
    'v.ssim_s',
    'v.class_acc_gap',
    'v.class_acc_im',
]


def train(valid_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_to_device(device)
    metrics = {field: list() for field in METRIC_FIELDS}
    # val
    _val(valid_data, metrics, device)
    fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    with open(f'./results/Sema_add2_test.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writerows([fit_metrics])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpp', type=float, default=1, help='bpp')
    parser.add_argument('--kernel_size1', type=int, default=3, help='kernel_size1')
    parser.add_argument('--kernel_size2', type=int, default=3, help='kernel_size2')
    parser.add_argument('--best_epoch', type=int, default=50, help='best_epoch')
    args = parser.parse_args()

    # argument
    best_epoch = args.best_epoch
    k1 = args.kernel_size1
    k2 = args.kernel_size2
    bpp = args.bpp
    H = 64
    W = 64
    if bpp < 1:
        bppk = 1
    else:
        bppk = bpp
    hidden_size = 16
    out_ch = 32
    #### NET ####
    # SC
    SC_encoder = sc_encoder(k1, k2, out_ch)
    SC_decoder = sc_decoder(k1, k2, out_ch)
    # classifier
    classifier = googlenet(3, 10)
    classifier.load_state_dict(torch.load('google_net_add.pkl'))
    # ste
    Ser_net = DenseEncoder(bppk, k1, k2, hidden_size, out_ch)
    Sdr_net = DenseDecoder(bppk, k1, k2, hidden_size, out_ch)
    # re-decode
    re_decoder = ReDecoder(bppk, hidden_size)

    SC_encoder = nn.DataParallel(SC_encoder)
    SC_decoder = nn.DataParallel(SC_decoder)
    classifier = nn.DataParallel(classifier)
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

    # load
    train_set = CIFAR10('./data', train=True, transform=data_tr_1, download=True)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = CIFAR10('./data', train=False, transform=data_tr_1, download=True)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    train(test_data)