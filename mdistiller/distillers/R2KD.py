import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from ._base import Distiller

def element_ranks(mat):
    batch_n, cls_n = mat.size()
    ranks = torch.Tensor.float(torch.argsort(torch.argsort(mat, dim=1), dim=1) + 1)
    div_n = cls_n*(cls_n + 1)/2
    return ranks/div_n

def cosine_similarity(a, b, eps=1e-8):
    return (a*b).sum(1) / (a.norm(p=2, dim=1)*b.norm(p=2, dim=1) + eps)

def spearman_correlation(pred_student, pred_teacher, eps=1e-8):
    pred_student_ranks = element_ranks(pred_student)
    pred_teacher_ranks = element_ranks(pred_teacher)
    return cosine_similarity(pred_student_ranks - pred_student_ranks.mean(1).unsqueeze(1), 
            pred_teacher_ranks - pred_teacher_ranks.mean(1).unsqueeze(1), eps)

def spearman_distance(stud, tea):
    return 1 - spearman_correlation(stud, tea).mean()

def rank_loss(logits_student, logits_teacher, temperature, rank_weight):
    pred_student = (logits_student / temperature).softmax(dim=1)
    pred_teacher = (logits_teacher / temperature).softmax(dim=1)
    spearman_loss = rank_weight * spearman_distance(pred_student, pred_teacher)
    return spearman_loss 

def abs_cosine_similarity(a, b, eps=1e-8):
    return torch.abs((a*b).sum(1)) / (a.norm(p=2, dim=1)*b.norm(p=2, dim=1) + eps)

def eisen_cosine_distance(stud, tea):
    return 1 - abs_cosine_similarity(stud, tea).mean()

def value_loss(logits_student, logits_teacher, temperature, value_weight):
    pred_student = (logits_student / temperature).softmax(dim=1)
    pred_teacher = (logits_teacher / temperature).softmax(dim=1)
    eisen_loss = value_weight*temperature**2*eisen_cosine_distance(pred_student, pred_teacher)
    eisen_loss += value_weight*temperature**2*eisen_cosine_distance(pred_student.transpose(0, 1), pred_teacher.transpose(0, 1))
    return eisen_loss 


class R2KD(Distiller):
    """Robustness-Reinforced Knowledge Distillation with Correlation Distance and Network Pruning
    (IEEE Transactions on Knowledge and Data Engineering 2024)"""

    def __init__(self, student, teacher, cfg, augmentation, pruned_teacher):
        super(R2KD, self).__init__(student, teacher)
        self.temperature = cfg.R2KD.TEMPERATURE
        self.value_loss_weight = cfg.R2KD.LOSS.VALUE_WEIGHT
        self.rank_loss_weight = cfg.R2KD.LOSS.RANK_WEIGHT

        self.pruned_teacher = pruned_teacher
        self.pruned_teacher_weight = cfg.PRUNE.WEIGHT

        self.aug = augmentation
        self.dataset = cfg.DATASET

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
            logits_pruned_teacher, _ = self.pruned_teacher(image)

        logits_teacher = self.pruned_teacher_weight * logits_teacher + (1.0 - self.pruned_teacher_weight) * logits_pruned_teacher

        # losses
        loss_ce = F.cross_entropy(logits_student, target)
        loss_value = value_loss(logits_student, logits_teacher, self.temperature, self.value_loss_weight)
        loss_rank = rank_loss(logits_student, logits_teacher, self.temperature, self.rank_loss_weight)
        
        if self.aug != 'NONE':

            if self.aug == 'mixup':
                mixed_image, original_target, mixing_target, lam = mixup_data(image, target)
            elif self.aug == 'cutmix':
                mixed_image, original_target, mixing_target, lam = cutmix_data(image, target)
            elif self.aug == 'cutout':
                mixed_image = cutout_data(image, target, self.dataset.TYPE)
            elif self.aug == 'autoaug':
                mixed_image = autoaug_data(image, target, self.dataset.TYPE)
            elif self.aug == 'cutmixpick':
                mixed_image, _, _ = cutmixpick_data(image, target, self.teacher)

            aug_logits_student, _ = self.student(mixed_image)
            with torch.no_grad():
                aug_logits_teacher, _ = self.teacher(mixed_image)
                aug_logits_pruned_teacher, _ = self.pruned_teacher(mixed_image)

            aug_logits_teacher = self.pruned_teacher_weight * aug_logits_teacher + (1.0 - self.pruned_teacher_weight) * aug_logits_pruned_teacher

            loss_value += value_loss(aug_logits_student, aug_logits_teacher, self.temperature, self.value_loss_weight)
            loss_rank += rank_loss(aug_logits_student, aug_logits_teacher, self.temperature, self.rank_loss_weight)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_value + loss_rank,
        }
        return logits_student, losses_dict

### Augmentation types ###

## MIXUP ##

def mixup_data(image, target, aug_alpha=0.2):
    lam = np.random.beta(aug_alpha, aug_alpha)
    index = torch.randperm(image.size(0)).cuda()
    mixed_image = lam*image + (1.0 - lam)*image[index]
    original_target, mixing_target = target, target[index]
    return mixed_image, original_target, mixing_target, lam

## CUTMIX ##

def cutmix_data(image, target, aug_alpha=1.0):
    image = image.clone()
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W*cut_rat)
        cut_h = np.int(H*cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
    lam = np.random.beta(aug_alpha, aug_alpha)
    index = torch.randperm(image.size(0)).cuda()
    original_target, mixing_target = target, target[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
    image[:, :, bbx1:bbx2, bby1:bby2] = image[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1)*(bby2 - bby1))/(image.size()[-1]*image.size()[-2])
    return image, original_target, mixing_target, lam

## CUTMIX_PICK ##

def get_entropy_batch(img, model_t):
    with torch.no_grad():
        logit, _ = model_t(img)
        prob = logit.softmax(dim=1)
        return (-prob * torch.log(prob)).sum(dim=1)

def cutmixpick_data(image, target, model_t):
    input_mix, original_target, mixing_target, lam = cutmix_data(image, target, 1.0)
    entropy_aug = get_entropy_batch(image, model_t)
    _, index = entropy_aug.sort()
    n_pick = min(len(index), 64)
    index = index[-n_pick:]

    rand_index, lam = 0, 0
    return input_mix[index], rand_index, lam

## CUTOUT ##

def cutout_data(image, target, dataset):
    n_holes = 1
    if dataset == 'cifar100':
        length = 8
    elif dataset == 'tiny_imagenet':
        length = 16

    _, _, h, w = image.size()
    mask = torch.ones((h,w)).cuda()
    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
    mask = mask.expand_as(image)
    input_mix = image * mask
    return input_mix

## AUTO_AUG ##

def denormalize_image(x, mean, std):
    '''x shape: [N, C, H, W], batch image'''
    x = x.cuda()
    mean = to_tensor(mean).cuda()
    std = to_tensor(std).cuda()
    mean = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3) # shape: [1, C, 1, 1]
    std = std.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    x = std * x + mean
    return x

def normalize_image(x, mean, std):
    '''x shape: [N, C, H, W], batch image'''
    x = x.cuda()
    mean = to_tensor(mean).cuda()
    std = to_tensor(std).cuda()
    mean = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3) # shape: [1, C, 1, 1]
    std = std.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    x = (x - mean) / std
    return x

def to_tensor(x):
    x = np.array(x)
    x = torch.from_numpy(x).float()
    return x

def autoaug_data(image, target, dataset):
    from torchvision.transforms.autoaugment import AutoAugment
    from torchvision.transforms.autoaugment import AutoAugmentPolicy

    if dataset == 'cifar100':
        policy = AutoAugmentPolicy.CIFAR10
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)

    elif dataset == 'tiny_imagenet':
        policy = AutoAugmentPolicy.IMAGENET
        MEAN = (0.485, 0.456, 0.406)
        STD = (0.229, 0.224, 0.225)

    autoaug = AutoAugment(policy=policy)

    image = denormalize_image(image, mean=MEAN, std=STD)
    image = (image.mul(255)).type(torch.uint8)
    input_mix = autoaug(image).type(torch.float32).div(255)
    input_mix = normalize_image(input_mix, mean=MEAN, std=STD)

    return input_mix

