from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
import torch


def _cosine_sim(a, b, eps=1e-6):
    prod_ab = torch.sum(a * b, dim=1)
    norm_a = torch.sum(a ** 2, dim=1).clamp(eps) ** 0.5
    norm_b = torch.sum(b ** 2, dim=1).clamp(eps) ** 0.5
    cos_sim = prod_ab / (norm_a * norm_b)
    return -cos_sim.mean()+1


def criterion1(classifier_model, x, label, raw):
    classifier_model.eval()
    z = classifier_model(x)  # class
    loss1 = nn.MSELoss()
    loss2 = nn.CrossEntropyLoss()
    loss = 50*loss1(x, raw) + loss2(z, label)
    return loss


def criterion2(imc, ims, im_s, im, lambda_list=None):
    if lambda_list is None:
        lambda_list = [1, 1, 1]
    loss1 = nn.MSELoss()
    loss = (lambda_list[0]*loss1(imc, ims) +
            lambda_list[1]*_cosine_sim(imc, ims) +
            lambda_list[2]*loss1(im_s, im))
    return loss


def criterion3(label_out, label_real):
    loss1 = nn.CrossEntropyLoss()
    loss = loss1(label_out, label_real)
    return loss


def criterion4(out, data):
    loss = binary_cross_entropy_with_logits(out, data)
    return loss


