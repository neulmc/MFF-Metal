import os
from os.path import join
import model
import model_tea
#import nestedunet
#import attu_net
import cv2
#import unet
from data_loader import preheat_Loader, our_preheat_Loader
from torch.utils.data import DataLoader
from utils import Logger, Averagvalue, save_checkpoint
from torchvision import transforms as transforms
import time
import sys
import metric
import shutil
import random
import numpy as np
import torch
from build_boundary import bound_loss
import torch.nn as nn
import torch.nn.functional as F


classes_num = 2

mode = 'OurModel'
name = 'Road-Dice-bs20-lr1e3-multi-jit'
dataset_dir = 'DATARoad'
TMP_DIR = mode + '_' + name


lr = 1e-3
batch_size = 20
maxepoch = 301
sche_gamma = 0.5
sche_miles = [100, 120, 150, 260]

bce_weight = 0  # 1
dice_weight = 1 #0.5  # 0.5
bound_weight = 0 #1e-4  # 0.01

model_block0 = 2
model_block1 = 2  #
model_block2 = 6  #
channel = [30,60,90,120]

jitter_d = 0.3 #0.3
jitter_p = 0.2 #0.2
random_c = 0


eval_freq = 10
print_freq = 10
m_c = 32

T_KD = False
T = 0.7
T_weight = 1

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 1029

mappings = dict()
mappings[1] = 255
mappings[2] = 64
mappings[3] = 128
mappings[4] = 100
mappings[5] = 200

multi_show = True

# tea_model
tea_block_1 = 3
tea_block_2 = 8
tea_tiny_c = 86

tea_channel = [180,360,540,720]

def load_teamodel():
    #tea_net = model_tea.self_net(block_1=tea_block_1, block_2=tea_block_2, tiny_c=tea_tiny_c)
    tea_net = model.self_net(block_0=model_block0, block_1=model_block1, block_2=model_block2, channel=tea_channel)
    #tea_path = "OurModel_tea-c120/epoch-250-checkpoint.pth"  #
    tea_path = "OurModel_tea-c180/epoch-140-checkpoint.pth"  #
    loaded_state_dict = torch.load(tea_path, map_location='cpu')
    model_state_dict = loaded_state_dict['state_dict']
    tea_net.load_state_dict(model_state_dict, strict=False)
    tea_net = tea_net.cuda()
    return tea_net


def seed_torch(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cross_entropy_loss(output, label, eps=1e-8):
    # one_hot_labels = torch.nn.functional.one_hot(label.long(), num_classes=4)
    # one_hot_labels = label.permute(0, 3, 1, 2)
    return - torch.sum(label * torch.log(torch.clamp(output, min=eps)))


def dice_loss(output, label, smooth=1):
    b, c, h, w = output.shape
    output1 = output[:, 1, :, :]
    label1 = label[:, 1, :, :]
    intersection1 = output1 * label1
    DSC1 = 1 - (2 * torch.abs(torch.sum(intersection1)) + smooth) / (torch.sum(output1) + torch.sum(label1) + smooth)
    return DSC1 * h * w / b

def train(model, train_loader, optimizer, epoch, batch_size):  # stu_net, tea_net,
    print('Traing Ep:%d' % epoch)
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()  #
    end = time.time()
    epoch_loss = []

    for i, (image, label, filename) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()
        _, c, w, h = image.shape

        outputs = model(image)
        loss = cross_entropy_loss(outputs, label) / batch_size
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses)
            print(info)

def train_selfnet(model, tea_model, train_loader, optimizer, epoch, batch_size):
    print('Traing Ep:%d' % epoch)
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    losses_bce = Averagvalue()
    losses_dice = Averagvalue()
    losses_dice1 = Averagvalue()
    losses_bou = Averagvalue()
    losses_bou1 = Averagvalue()
    losses_tea = Averagvalue()
    # switch to train mode

    tea_model.eval()  #
    model.train()  # model

    # model.train()
    end = time.time()
    epoch_loss = []

    for i, (image, label, label_bou, filename) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label, label_bou = image.cuda(), label.cuda(), label_bou.cuda()
        _, c, w, h = image.shape
        if T_KD:
            y_student = model(image, T_KD = True)
            outputs = F.softmax(y_student)
            y_teacher = tea_model(image, T_KD = True)  #
            y_teacher = y_teacher.detach()  # 切掉反向传播
            loss_KD = nn.KLDivLoss(reduction='sum')  #
            loss_dis = loss_KD(F.log_softmax(y_student / T, dim=1), F.softmax(y_teacher / T, dim=1)) / batch_size  #
        else:
            outputs = model(image)
            loss_dis = torch.zeros(1).cuda()

        bce_loss = cross_entropy_loss(outputs, label) / batch_size

        dice1 = dice_loss(outputs, label)
        bound1  = bound_loss(outputs, label_bou)

        loss = bce_loss * bce_weight + (dice1) * dice_weight + (
                    bound1) * bound_weight + T_weight * loss_dis * batch_size
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        losses_bce.update(bce_loss.item(), image.size(0))
        losses_dice.update((dice1).item(), image.size(0))
        losses_dice1.update(dice1.item(), image.size(0))
        losses_bou.update((bound1 ).item(), image.size(0))
        losses_bou1.update(bound1.item(), image.size(0))
        losses_tea.update(loss_dis.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=losses) + \
                   'DisLoss {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=losses_tea) + \
                   'BCELoss {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=losses_bce) + \
                   'DiceLoss {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=losses_dice) + \
                   'BoundLoss {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=losses_bou) + \
                   '(dice1 avg:{loss.avg:.2f} '.format(loss=losses_dice1) + \
                   '(bou1 avg:{loss.avg:.2f} '.format(loss=losses_bou1)
            print(info)


def save(model, epoch):
    # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }, filename=join(TMP_DIR, "epoch-%d-checkpoint.pth" % epoch))


def test(model, test_loader, epoch):
    def build_predictions(outputs, label):
        pred = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0]
        prob = outputs[0].permute(1, 2, 0).detach().cpu().numpy()
        pred_show = pred.copy()
        pred_show[pred == 1] = mappings[1]
        lab = torch.argmax(label, dim=1).detach().cpu().numpy()[0]
        lab_show = lab.copy()
        lab_show[lab == 1] = mappings[1]
        return pred, prob, pred_show, lab_show, lab

    print('Testing')
    save_dir = TMP_DIR + '/' + str(epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.eval()
    for i, (image, label, filename) in enumerate(test_loader):
        image, label = image.cuda(), label.cuda()
        filename = filename[0]
        if multi_show and mode == 'OurModel':
            outputs, [o48,o24], gp = model(image, multi_show=True)
            image_np = image.detach().cpu().numpy()[0]
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            pred, prob, pred_show, lab_show, lab = build_predictions(outputs, label)
            _, _, pred_showo48, _, _ = build_predictions(o48, label)
            _, _, pred_showo24, _, _ = build_predictions(o24, label)
            imggt = np.hstack([image_np, lab_show, pred_show])
            multipred = np.hstack([pred_showo48, pred_showo24])
            gp = gp.detach().cpu().numpy()[0, :, 0, 0]
            cv2.imwrite(save_dir + '/' + filename + '.png', imggt)
            cv2.imwrite(save_dir + '/' + filename +
                        '_multipred(48-' + str(round(gp[0], 2)) + '_24-' + str(round(gp[1], 2)) + ')' + '.png', multipred)
            np.save(save_dir + '/' + filename + '_prob', prob)
        else:
            outputs = model(image)
            image_np = image.detach().cpu().numpy()[0]
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            pred, prob, pred_show, lab_show, lab = build_predictions(outputs, label)
            imggt = np.hstack([image_np, lab_show, pred_show])
            cv2.imwrite(save_dir + '/' + filename + '.png', imggt)
            np.save(save_dir + '/' + filename + '_prob', prob)

# EVAL
def eval(test_loader, epoch):
    print('Eval ' + mode)
    load_dir = TMP_DIR + '/' + str(epoch)
    iou1_fenzi = 0
    iou1_fenmu = 0

    for i, (image, label, filename) in enumerate(test_loader):
        filename = filename[0] + '.png'
        imggt = cv2.imread(load_dir + '/' + filename, 0)
        h, w = imggt.shape
        w_ = w / 3
        gt = imggt[:, int(w_):int(w_ * 2)]
        pred = imggt[:, int(w_ * 2):]
        gt[gt == mappings[1]] = 1
        pred[pred == mappings[1]] = 1
        prob = np.load(load_dir + '/' + filename.replace('.png', '_prob.npy'))
        iou_fenzi, iou_fenmu = metric.compute_accurate(pred.copy(), gt.copy())
        iou1_fenzi += iou_fenzi[0]
        iou1_fenmu += iou_fenmu[0]

    dice1 = iou1_fenzi / iou1_fenmu

    print('IoU1: ' + str(dice1))
    print('mIoU: ' + str((dice1 ) / 1))


if __name__ == '__main__':
    seed_torch()
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    shutil.copy(os.path.abspath(__file__), TMP_DIR + '/config')
    # data
    if mode == 'OurModel':
        train_dataset = our_preheat_Loader(root=dataset_dir, split="train", jitter_d=jitter_d, jitter_p=jitter_p,
                                           random_c=random_c)
        test_dataset = our_preheat_Loader(root=dataset_dir, split="test", )
        tea_model = load_teamodel()
    else:
        train_dataset = preheat_Loader(root=dataset_dir, split="train", )
        test_dataset = preheat_Loader(root=dataset_dir, split="test", )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=1, drop_last=True, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        num_workers=1, drop_last=False, shuffle=False)

    if mode == 'OurModel':
        model = model.self_net(classes = classes_num, block_0=model_block0, block_1=model_block1, block_2=model_block2, channel = channel)
    elif mode == 'Unet':
        model = unet.UNet(model_c=m_c)
    elif mode == 'NestedUNet':
        model = nestedunet.NestedUNet(num_classes=4)
    elif mode == 'AttU_Net':
        model = attu_net.AttU_Net(output_ch=4, c=m_c)
    elif mode == 'transnet':
        model = transunet.TransUNet(img_dim=200,
                                    in_channels=3,
                                    out_channels=64,
                                    head_num=4,
                                    mlp_dim=1024,
                                    block_num=13,
                                    patch_dim=16, class_num=4)
    elif mode == 'TeaModel':
        model = model_tea.self_net(block_1=tea_block_1, block_2=tea_block_2, tiny_c=tea_tiny_c)
    model.cuda()

    # log
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' % ('adam', lr)))
    sys.stdout = log

    num_parameters = sum(p.numel() for p in model.parameters())  # 统计参数数量
    print(f"parameters_num: {num_parameters / 1e3}" + 'k')  # 输出参数数量
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sche_miles, gamma=sche_gamma)
    for epoch in range(maxepoch):
        # train
        print('lr: ' + str(optimizer.param_groups[0]['lr']))
        if mode == 'OurModel':
            train_selfnet(model, tea_model, train_loader, optimizer, epoch, batch_size)
        else:
            train(model, train_loader, optimizer, epoch, batch_size)
        if epoch % eval_freq == 0 and epoch != 0:
            timestart = time.time()
            test(model, test_loader, epoch)
            timeend = time.time()
            print('FPS: ' + str(len(test_loader) / (timeend - timestart)))
            eval(test_loader, epoch)
            save(model, epoch)  # save model
        log.flush()  # write log
        scheduler.step()