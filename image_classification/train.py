import gc
import csv
import argparse
from tqdm import tqdm
from torch import nn
import torch.utils.data
from torchvision.datasets import CIFAR10
from torch.nn.functional import mse_loss

from Models.Criterion import _cosine_sim, criterion4, criterion3
from Models.RED_CNN_add import sc_encoder, sc_decoder, Quantization
from Models.GoogleNet import googlenet
from Models.famo import FAMO
from Models.SD_model_add import DenseEncoder, DenseDecoder, SRNet, ReDecoder
from Models.utils_s import _ssim, get_acc, data_tr_1, sr_Net_data, get_payload, get_payload01


def net_to_device(device):
    SC_encoder.to(device)
    SC_decoder.to(device)
    classifier.to(device)
    critic1.to(device)
    critic2.to(device)
    Ser_net.to(device)
    Sdr_net.to(device)
    re_decoder.to(device)


def _transmit(im, payload, train=False, shuffle=False):
    imc = SC_encoder(im)
    ims = Ser_net(imc.detach(), payload.detach())
    if shuffle:
        idx = torch.randperm(ims.shape[0])
        ims = ims[idx, :]
    imc_max = torch.round(torch.max(torch.abs(imc)))
    imc = imc / imc_max
    ims = ims / imc_max
    if train:
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


def _receiver(imc, ims, train=False):
    out_data = Sdr_net(ims)

    im_s = SC_decoder(ims)
    im_c = SC_decoder(imc)

    if train:
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


def _fit_critic(train_data, device):
    gc.collect()
    critic1.train()
    critic2.train()
    SC_encoder.eval()
    SC_decoder.eval()
    Ser_net.eval()
    pbar = tqdm(train_data)
    for im, _ in pbar:
        im = im.to(device)
        if bpp < 1:
            payload01 = torch.zeros((im.shape[0], 1, 1, round(bpp * H * W))).random_(0, 2).to(device)
            payload = get_payload(payload01, bpp, H, W, device)
        else:
            payload = torch.zeros((im.shape[0], int(bpp), H, W)).random_(0, 2).to(device)

        imc, ims = _transmit(im, payload, True, False)
        ##### channel #######
        im_c, im_s, _, _ = _receiver(imc, ims, True)

        # score
        im_train, label_train = sr_Net_data(imc.float(), ims.float())  # [0:l]
        im_train = im_train.to(device)
        label_train = label_train.to(device)
        label_out = critic1(im_train)
        loss1 = criterion3(label_out, label_train)

        im_train, label_train = sr_Net_data(im_c.float(), im_s.float())  # [0:l]
        im_train = im_train.to(device)
        label_train = label_train.to(device)
        label_out = critic2(im_train)
        loss2 = criterion3(label_out, label_train)

        loss = loss1 + loss2
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()
        pbar.set_description(
            f"Loss1: {loss1.item():.5f}; Loss2: {loss2.item():.5f}"
        )
    return loss.item()


def _fit0(train_data, lr, weight_opt, device):
    optimizer = torch.optim.Adam(dec_list, lr=lr)
    gc.collect()
    Ser_net.train()
    Sdr_net.train()
    SC_encoder.train()
    SC_decoder.train()
    re_decoder.train()
    critic1.eval()
    critic2.eval()
    classifier.eval()
    pbar = tqdm(train_data)
    for im, _ in pbar:
        im = im.to(device)
        if bpp < 1:
            payload01 = torch.zeros((im.shape[0], 1, 1, round(bpp * H * W))).random_(0, 2).to(device)
            payload = get_payload(payload01, bpp, H, W, device)
        else:
            payload = torch.zeros((im.shape[0], int(bpp), H, W)).random_(0, 2).to(device)

        imc = SC_encoder(im)
        ims = Ser_net(imc.detach(), payload.detach())
        ##### channel #######
        out_data = Sdr_net(ims)
        im_s = SC_decoder(ims)
        out_data2 = re_decoder(im_s)

        # score
        loss2_1 = mse_loss(imc, ims)
        loss2_2 = _cosine_sim(imc, ims)
        loss4_1 = mse_loss(im_s, im)
        if bpp < 1:
            loss1 = criterion4(get_payload01(out_data, bpp, H, W, device), payload01)
            loss4 = criterion4(get_payload01(out_data2, bpp, H, W, device), payload01)
        else:
            loss1 = criterion4(out_data, payload)
            loss4 = criterion4(out_data2, payload)
        loss2 = torch.cat([
            torch.unsqueeze(loss4_1 + loss2_1, 0),
            torch.unsqueeze(loss1, 0),
            torch.unsqueeze(loss4, 0),
        ])

        optimizer.zero_grad()
        weight_opt.backward(loss2)
        pbar.set_description(
            f"Loss1: {loss1.item():.5f}; Loss4: {loss4.item():.5f} 4_1 {loss4_1.item():.5f}  mse: {loss2_1.item():.5f}"
        )
        optimizer.step()

        with torch.no_grad():
            imc = SC_encoder(im)
            ims = Ser_net(imc.detach(), payload.detach())
            ##### channel #######
            out_data = Sdr_net(ims)
            im_s = SC_decoder(ims)
            out_data2 = re_decoder(im_s)

            # score
            loss2_1 = mse_loss(imc, ims)
            loss4_1 = mse_loss(im_s, im)
            if bpp < 1:
                loss1 = criterion4(get_payload01(out_data, bpp, H, W, device), payload01)
                loss4 = criterion4(get_payload01(out_data2, bpp, H, W, device), payload01)
            else:
                loss1 = criterion4(out_data, payload)
                loss4 = criterion4(out_data2, payload)
            new_loss = torch.cat([
                torch.unsqueeze(loss4_1 + loss2_1, 0),
                torch.unsqueeze(loss1, 0),
                torch.unsqueeze(loss4, 0),
            ])
            weight_opt.update(new_loss)


def _fit_sc(train_data, lr, weight_opt, device):
    optimizer = torch.optim.Adam(dec_list, lr=lr)
    gc.collect()
    Ser_net.train()
    Sdr_net.train()
    SC_encoder.train()
    SC_decoder.train()
    re_decoder.train()
    classifier.eval()
    pbar = tqdm(train_data)
    for im, label in pbar:
        im = im.to(device)
        label = label.to(device)
        if bpp < 1:
            payload01 = torch.zeros((im.shape[0], 1, 1, round(bpp * H * W))).random_(0, 2).to(device)
            payload = get_payload(payload01, bpp, H, W, device)
        else:
            payload = torch.zeros((im.shape[0], int(bpp), H, W)).random_(0, 2).to(device)

        imc = SC_encoder(im)
        ims = Ser_net(imc.detach(), payload.detach())
        ##### channel #######
        out_data = Sdr_net(ims)
        im_s = SC_decoder(ims)
        im_c = SC_decoder(imc)
        out_data2 = re_decoder(im_s)

        # score
        loss2_1 = mse_loss(imc, ims)
        loss4_1 = mse_loss(im_s, im)
        # loss3_1 = mse_loss(im_c, im)
        # loss3_2 = nn.CrossEntropyLoss()(classifier(im_c), label)
        loss4_2 = nn.CrossEntropyLoss()(classifier(im_s), label)
        if bpp < 1:
            loss1 = criterion4(get_payload01(out_data, bpp, H, W, device), payload01)
            loss4 = criterion4(get_payload01(out_data2, bpp, H, W, device), payload01)
        else:
            loss1 = criterion4(out_data, payload)
            loss4 = criterion4(out_data2, payload)
        loss2 = torch.cat([
            torch.unsqueeze(loss4_1 + 0.01 * loss4_2, 0),
            torch.unsqueeze(loss1 + loss2_1, 0),
            torch.unsqueeze(loss4, 0),
        ])

        optimizer.zero_grad()
        weight_opt.backward(loss2)
        pbar.set_description(
            f"Loss1: {loss1.item():.5f} Loss4: {loss4.item():.5f} 2_1:{loss2_1.item():.5f} 4_2: {loss4_2.item():.5f}; 4_1:{loss4_1.item():.5f}"
        )
        optimizer.step()

        with torch.no_grad():
            imc = SC_encoder(im)
            ims = Ser_net(imc.detach(), payload.detach())
            ##### channel #######
            out_data = Sdr_net(ims)
            im_s = SC_decoder(ims)
            im_c = SC_decoder(imc)
            out_data2 = re_decoder(im_s)
            # score
            loss2_1 = mse_loss(imc, ims)
            loss4_1 = mse_loss(im_s, im)
            # loss3_1 = mse_loss(im_c, im)
            # loss3_2 = nn.CrossEntropyLoss()(classifier(im_c), label)
            loss4_2 = nn.CrossEntropyLoss()(classifier(im_s), label)
            if bpp < 1:
                loss1 = criterion4(get_payload01(out_data, bpp, H, W, device), payload01)
                loss4 = criterion4(get_payload01(out_data2, bpp, H, W, device), payload01)
            else:
                loss1 = criterion4(out_data, payload)
                loss4 = criterion4(out_data2, payload)
            new_loss = torch.cat([
                torch.unsqueeze(loss4_1 + 0.01 * loss4_2, 0),
                torch.unsqueeze(loss1 + loss2_1, 0),
                torch.unsqueeze(loss4, 0),
            ])
            weight_opt.update(new_loss)


def _fit3(train_data, lr, weight_opt, device):
    optimizer = torch.optim.Adam(dec_list, lr=lr)
    gc.collect()
    Ser_net.train()
    Sdr_net.train()
    SC_encoder.train()
    SC_decoder.train()
    re_decoder.train()
    critic1.eval()
    critic2.eval()
    classifier.eval()
    pbar = tqdm(train_data)
    for im, label in pbar:
        with torch.no_grad():
            im = im.to(device)
            label = label.to(device)
            if bpp < 1:
                payload01 = torch.zeros((im.shape[0], 1, 1, round(bpp * H * W))).random_(0, 2).to(device)
                payload = get_payload(payload01, bpp, H, W, device)
            else:
                payload = torch.zeros((im.shape[0], int(bpp), H, W)).random_(0, 2).to(device)

        imc, ims = _transmit(im, payload, True, False)
        ##### channel #######
        im_c, im_s, out_data, out_data2 = _receiver(imc, ims, True)

        # score
        im_train, label_train = sr_Net_data(imc.float(), ims.float())
        im_train = im_train.to(device)
        label_train = label_train.to(device)
        label_out = critic1(im_train)
        sr_loss = criterion3(label_out, label_train)

        im_train, label_train = sr_Net_data(im_c.float(), im_s.float())
        im_train = im_train.to(device)
        label_train = label_train.to(device)
        label_out = critic2(im_train)
        sr_loss2 = criterion3(label_out, label_train)

        if bpp < 1:
            loss1 = criterion4(get_payload01(out_data, bpp, H, W, device), payload01)
            loss4 = criterion4(get_payload01(out_data2, bpp, H, W, device), payload01)
        else:
            loss1 = criterion4(out_data, payload)
            loss4 = criterion4(out_data2, payload)

        imc_max = torch.round(torch.max(torch.abs(imc)))
        loss2_1 = mse_loss(imc / imc_max, ims / imc_max)
        loss2_2 = _cosine_sim(imc / imc_max, ims / imc_max)
        loss2_3 = mse_loss(im_c, im_s)
        loss3_1 = mse_loss(im_c, im)
        loss3_2 = nn.CrossEntropyLoss()(classifier(im_c), label)
        loss4_1 = mse_loss(im_s, im)
        loss4_2 = nn.CrossEntropyLoss()(classifier(im_s), label)

        loss2 = torch.cat([
            torch.unsqueeze(loss3_1 + 0.01 * loss3_2, 0),
            torch.unsqueeze(loss4_1 + 0.01 * loss4_2, 0),
            torch.unsqueeze(loss2_1, 0),
            torch.unsqueeze(loss2_2, 0),
            torch.unsqueeze(loss2_3, 0),
            torch.unsqueeze(loss1 + loss2_1 + loss4_1, 0),
            torch.unsqueeze(loss4 + loss2_3 + loss4_1, 0),
            torch.unsqueeze(1 / (sr_loss + 1e-6), 0),
            torch.unsqueeze(1 / (sr_loss2 + 1e-6), 0),
        ])
        optimizer.zero_grad()
        weight_opt.backward(loss2)
        if bpp < 1:
            decoder_acc = (get_payload01(out_data, bpp, H, W, device) >= 0.0).eq(
                payload01 >= 0.5).sum().float() / payload01.numel()  # M-/M
            decoder_acc2 = (get_payload01(out_data2, bpp, H, W, device) >= 0.0).eq(
                payload01 >= 0.5).sum().float() / payload01.numel()  # M-/M
        else:
            decoder_acc = (out_data >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()  # M-/M
            decoder_acc2 = (out_data2 >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()  # M-/M
        psnr2 = 10 * torch.log10(4 / mse_loss(im_c, im_s))
        psnr3 = 10 * torch.log10(4 / mse_loss(im_s, im))
        pbar.set_description(
            f"Loss1: {decoder_acc.item():.3f}; Loss4: {decoder_acc2.item():.3f} psnr2: {psnr2.item():.2f} "
            f"sr_loss:{sr_loss.item():.5f} Loss4_1: {psnr3.item():.5f} "
            f"mse: {loss2_1.item():.5f} {loss2_2.item():.5f} {loss2_3.item():.5f}"
        )
        optimizer.step()

        with torch.no_grad():
            imc, ims = _transmit(im, payload, True, False)
            ##### channel #######
            im_c, im_s, out_data, out_data2 = _receiver(imc, ims, True)
            # score
            im_train, label_train = sr_Net_data(imc.float(), ims.float())
            im_train = im_train.to(device)
            label_train = label_train.to(device)
            label_out = critic1(im_train)
            sr_loss = criterion3(label_out, label_train)

            im_train, label_train = sr_Net_data(im_c.float(), im_s.float())
            im_train = im_train.to(device)
            label_train = label_train.to(device)
            label_out = critic2(im_train)
            sr_loss2 = criterion3(label_out, label_train)

            if bpp < 1:
                loss1 = criterion4(get_payload01(out_data, bpp, H, W, device), payload01)
                loss4 = criterion4(get_payload01(out_data2, bpp, H, W, device), payload01)
            else:
                loss1 = criterion4(out_data, payload)
                loss4 = criterion4(out_data2, payload)

            imc_max = torch.round(torch.max(torch.abs(imc)))
            loss2_1 = mse_loss(imc / imc_max, ims / imc_max)
            loss2_2 = _cosine_sim(imc / imc_max, ims / imc_max)
            loss2_3 = mse_loss(im_c, im_s)
            loss3_1 = mse_loss(im_c, im)
            loss3_2 = nn.CrossEntropyLoss()(classifier(im_c), label)
            loss4_1 = mse_loss(im_s, im)
            loss4_2 = nn.CrossEntropyLoss()(classifier(im_s), label)

            new_loss = torch.cat([
                torch.unsqueeze(loss3_1 + 0.01 * loss3_2, 0),
                torch.unsqueeze(loss4_1 + 0.01 * loss4_2, 0),
                torch.unsqueeze(loss2_1, 0),
                torch.unsqueeze(loss2_2, 0),
                torch.unsqueeze(loss2_3, 0),
                torch.unsqueeze(loss1 + loss2_1 + loss4_1, 0),
                torch.unsqueeze(loss4 + loss2_3 + loss4_1, 0),
                torch.unsqueeze(1 / (sr_loss + 1e-6), 0),
                torch.unsqueeze(1 / (sr_loss2 + 1e-6), 0),
            ])
            weight_opt.update(new_loss)


def _val(validate, metrics, device):
    gc.collect()
    Ser_net.eval()
    Sdr_net.eval()
    SC_encoder.eval()
    SC_decoder.eval()
    critic1.eval()
    critic2.eval()
    classifier.eval()
    re_decoder.eval()
    pbar = tqdm(validate)
    for im, label in pbar:
        with torch.no_grad():
            im = im.to(device)
            label = label.to(device)
            if bpp < 1:
                payload01 = torch.zeros((im.shape[0], 1, 1, round(bpp * H * W))).random_(0, 2).to(device)
                payload = get_payload(payload01, bpp, H, W, device)
            else:
                payload = torch.zeros((im.shape[0], int(bpp), H, W)).random_(0, 2).to(device)  # 64
            imc, ims = _transmit(im, payload, False, False)
            ##### channel #######
            im_c, im_s, out_data, out_data2 = _receiver(imc, ims, False)

            if bpp < 1:
                decoder_acc = (get_payload01(out_data, bpp, H, W, device) >= 0.0).eq(
                    payload01 >= 0.5).sum().float() / payload01.numel()  # M-/M
                decoder_acc2 = (get_payload01(out_data2, bpp, H, W, device) >= 0.0).eq(
                    payload01 >= 0.5).sum().float() / payload01.numel()  # M-/M
            else:
                decoder_acc = (out_data >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()  # M-/M
                decoder_acc2 = (out_data2 >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()  # M-/M

            acc_im = get_acc(classifier(im), label)
            acc_s = get_acc(classifier(im_s), label)
            imc_max = torch.round(torch.max(torch.abs(imc)))
        pbar.set_description(
            f"Loss1: {decoder_acc.item():.3f}; Loss4: {decoder_acc2.item():.3f}"
        )
        metrics['v.de_acc'].append(decoder_acc.item())
        metrics['v.de_acc2'].append(decoder_acc2.item())

        metrics['v.psnr1'].append(10 * torch.log10(4 / mse_loss(imc / imc_max, ims / imc_max)).item())
        metrics['v.ssim1'].append(_ssim(im_c / imc_max, im_s / imc_max, 2).item())
        metrics['v.psnr2'].append(10 * torch.log10(4 / mse_loss(im_c, im_s)).item())
        metrics['v.ssim2'].append(_ssim(im_c, im_s, 2).item())

        metrics['v.psnr_s'].append(10 * torch.log10(4 / mse_loss(im_s, im)).item())
        metrics['v.ssim_s'].append(_ssim(im_s, im, 2).item())
        metrics['v.class_acc_gap'].append(acc_im - acc_s)
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


def train(train_data, valid_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_to_device(device)
    weight_opt_sc = FAMO(n_tasks=3, device=device)
    weight_opt0 = FAMO(n_tasks=3, device=device)
    weight_opt3 = FAMO(n_tasks=9, device=device)
    metrics = {field: list() for field in METRIC_FIELDS}
    with open(f'./Sema/Sema_add2_{bpp}{k1}{k2}.csv', 'a', newline='', encoding='utf-8') as f:
        if not continue_train:
            f.truncate(0)
    if continue_train:
        checkpoint = torch.load(
            f'./model/SD_add2{bpp}{k1}{k2}_%s.pth' % (str(continue_epoch)))  # 加载断点
        SC_encoder.load_state_dict(checkpoint['SC_encoder'])
        SC_decoder.load_state_dict(checkpoint['SC_decoder'])
        Ser_net.load_state_dict(checkpoint['Ser_net'])
        Sdr_net.load_state_dict(checkpoint['Sdr_net'])
        re_decoder.load_state_dict(checkpoint['re_decoder'])
        critic1.load_state_dict(checkpoint['critic1'])
        critic2.load_state_dict(checkpoint['critic2'])
        start_epoch = continue_epoch
    else:
        start_epoch = -1
    lr = 1e-3
    for epoch in range(start_epoch + 1, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        # train
        if lr > 1e-5 and (epoch + 1) % 30 == 0:
            lr = lr / 10
            print(lr)
        if epoch == 0:
            _fit0(train_data, lr, weight_opt0, device)
            _fit_sc(train_data, lr, weight_opt_sc, device)
        else:
            l = _fit_critic(train_data, device)
            while l > 1:
                print(l)
                l = _fit_critic(train_data, device)
            _fit3(train_data, lr, weight_opt3, device)
        # val
        _val(valid_data, metrics, device)
        fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        with open(f'./Sema/Sema_add2_{bpp}{k1}{k2}.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
            writer.writerows([fit_metrics])

        checkpoint = {
            "SC_encoder": SC_encoder.state_dict(),
            "SC_decoder": SC_decoder.state_dict(),
            "Ser_net": Ser_net.state_dict(),
            "Sdr_net": Sdr_net.state_dict(),
            "re_decoder": re_decoder.state_dict(),
            "critic1": critic1.state_dict(),
            "critic2": critic2.state_dict(),
        }
        torch.save(checkpoint, f'./model/SD_add2{bpp}{k1}{k2}_%s.pth' % (str(epoch)))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpp', type=float, default=1, help='bpp')
    parser.add_argument('--kernel_size1', type=int, default=3, help='kernel_size1')
    parser.add_argument('--kernel_size2', type=int, default=3, help='kernel_size2')
    parser.add_argument('--num_epochs', type=int, default=50, help='num_epochs')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue to train')
    parser.add_argument('--continue_epoch', type=int, default=0, help='the epoch continue to train')
    args = parser.parse_args()

    # argument
    continue_train = args.continue_train
    continue_epoch = args.continue_epoch
    bpp = args.bpp
    k1 = args.kernel_size1
    k2 = args.kernel_size2
    num_epochs = args.num_epochs
    H = 64
    W = 64
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
    Ser_net = DenseEncoder(bpp, k1, k2, hidden_size, out_ch)
    Sdr_net = DenseDecoder(bpp, k1, k2, hidden_size, out_ch)
    critic1 = SRNet()
    critic2 = SRNet()
    # re-decode
    re_decoder = ReDecoder(bpp, hidden_size)

    SC_encoder = nn.DataParallel(SC_encoder)
    SC_decoder = nn.DataParallel(SC_decoder)
    classifier = nn.DataParallel(classifier)
    Ser_net = nn.DataParallel(Ser_net)
    Sdr_net = nn.DataParallel(Sdr_net)
    re_decoder = nn.DataParallel(re_decoder)
    critic1 = nn.DataParallel(critic1)
    critic2 = nn.DataParallel(critic2)

    # optimizer
    critic_optimizer = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=1e-3)
    dec_list = list(
        SC_encoder.parameters()) + list(
        Ser_net.parameters()) + list(
        SC_decoder.parameters()) + list(
        re_decoder.parameters()) + list(
        Sdr_net.parameters())

    # load
    train_set = CIFAR10('./data', train=True, transform=data_tr_1, download=True)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = CIFAR10('./data', train=False, transform=data_tr_1, download=True)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    train(train_data, test_data)
