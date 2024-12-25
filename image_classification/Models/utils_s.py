import zlib
from skimage.metrics import structural_similarity as ssim
import torch
from reedsolo import RSCodec
import numpy as np
from PIL import Image
import math
rs = RSCodec(250)


def get_payload(payload01, bpp, H, W, device):
    if bpp < 1:
        payload02 = payload01
        l = round(bpp * H * W)
        c = math.floor((H * W - l) / l)
        while c > 0:
            payload02 = torch.cat([payload02, payload01], dim=3)
            c = c - 1
        a = payload01[:, :, :, :H * W - l - math.floor((H * W - l) / l) * l]
        payload02 = torch.cat([payload02, a], dim=3)
        payload = payload02.reshape(payload02.shape[0], 1, H, W).to(device)
    else:
        payload = payload01.reshape(payload01.shape[0], 1, H, W).to(device)
    return payload


def get_payload01(out_data, bpp, H, W, device):
    if bpp < 1:
        payload02 = out_data.reshape(out_data.shape[0], 1, 1, H*W)
        l = round(bpp * H * W)
        c = math.floor((H * W - l) / l)
        payload01 = payload02[:, :, :, 0:l]
        i = 1
        while c > 0:
            b = payload02[:, :, :, i*l:i*l+l]
            payload01 = payload01 + b
            i = i + 1
            c = c - 1
        a = torch.zeros(out_data.shape[0], 1, 1, l).to(device)
        a[:, :, :, :H * W - l - math.floor((H * W - l) / l) * l] = payload02[:, :, :, math.floor((H * W - l) / l)*l+l:H*W]
        payload01 = payload01 + a
        payload01 = payload01/(math.floor((H * W - l) / l)+2)
    else:
        payload01 = out_data.reshape(out_data.shape[0], 1, 1, H*W).to(device)
    return payload01


def save_pic(im, name):
    im_pic = im #im[0]
    im_pic = im_pic.permute(1, 2, 0)
    im_pic = im_pic.detach().cpu().numpy()
    im_pic = np.array(im_pic, dtype='float32')
    im_pic = (im_pic * 0.5 + 0.5) * 255
    #im_pic = np.ceil(im_pic).astype('int')
    im_pic = np.uint8(im_pic)
    im_pic = Image.fromarray(im_pic)
    im_pic.save(name)


def data_tr(x):
    x = x.resize((360, 360))
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x


def data_tr_1(x):
    x = x.resize((64, 64))
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x


def _ssim(cover, generated, data_range):
    SSIMi = torch.empty((1, cover.shape[0]))
    for i in range(cover.shape[0]):
        c = cover[i]
        c = c.permute(1, 2, 0)
        c = c.detach().cpu().numpy()
        g = generated[i]
        g = g.permute(1, 2, 0)
        g = g.detach().cpu().numpy()
        SSIMi[0, i] = ssim(c, g, data_range=data_range, multichannel=True)
    return SSIMi.mean(1)


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def _random_data(cover, data_depth):
    N, _, H, W = cover.size()
    plo = torch.zeros((N, data_depth, H, W)).random_(0, 2)
    return plo


def sr_Net_data(imc,ims):
    imc_length = imc.shape[0]
    ims_length = ims.shape[0]
    data = [0 for _ in range(imc_length)]
    label_c = torch.tensor(data, dtype=torch.long)
    data = [1 for _ in range(ims_length)]
    label_s = torch.tensor(data, dtype=torch.long)
    l = np.arange(0, imc_length + ims_length)
    np.random.shuffle(l)
    label_train = torch.cat([label_c, label_s], dim=0)
    label_train = label_train[l]
    im_train = torch.cat([imc, ims], dim=0)
    im_train = im_train[l,]
    return im_train,label_train


def get_Jpeg(J2, q):
    y = J2
    for i in range(J2.shape[0]):
        J2i = J2[i].permute(1, 2, 0)
        J2i = J2i.detach().cpu().numpy()
        J2i = np.array(J2i, dtype='int')
        J2i = np.uint8(J2i)
        J2i = Image.fromarray(J2i)
        J2i.save('test_JPEG.jpg', quality = q)
        #J2i = eng.imread('test_JPEG.jpg')
        #eng.imwrite(J2i, 'test_JPEG.jpg', 'jpg', 'mode', 'lossy', 'quality', q, nargout=0)
        yy = cv2.imread('test_JPEG.jpg')
        yy = torch.from_numpy(yy)
        y[i] = yy.permute(2, 0, 1)
    return y