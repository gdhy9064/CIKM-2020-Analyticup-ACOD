
import sys
# use modified packages
sys.path.insert(0, './eval_code')
sys.path.insert(0, './mmdetection')

import cv2
import json
import matplotlib.pyplot as plt
import mmcv
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.apis import init_detector, inference_detector
from mmdet.apis.inference import LoadImage
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
import numpy as np
import os
import pickle
from PIL import ImageFile
import random
from tool.darknet2pytorch import *
import torch
from torchvision import transforms
from utils.utils import *
import warnings


ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

lst = os.listdir('eval_code/select1000_new/images/')
lst.sort()
def load_yolov4():
    cfgfile = "eval_code/models/yolov4.cfg"
    weightfile = "eval_code/models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    for param in darknet_model.parameters():
        param.requires_grad = False
    return darknet_model

yolo = load_yolov4()

config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint = './eval_code/models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    
rcnn = init_detector(config, checkpoint, device='cuda:0')
for param in rcnn.parameters():
    param.requires_grad = False


# 贴图训练模型
class AdversarialPatch(torch.nn.Module):
    def __init__(self, yolo, rcnn, yxs, patches_shape, ratio):
        '''
        :yolo: yolo model
        :rcnn: faster-rcnn model
        :yxs: center parameters of lines of net-like patch, a list of (y, x)
        :patchs_shape: a list of (h, w) of lines
        :ratio: trade-off between faster-rcnn and yolo，loss = rcnn_loss * ratio + yolo_loss
        '''
        super().__init__()
        self.yolo = yolo
        self.rcnn = rcnn
        self.yxs = yxs
        self.hws = patches_shape
        self.ratio = ratio
        # line patches
        self.patches = [torch.zeros([3, *patch_shape]).cuda().requires_grad_(True) for patch_shape in patches_shape]

    def forward(self, data, img):
        '''
        :data: the structured input of faster-rcnn
        :img: image tensor of shape (1, 3, 500, 500), ranging from 0 to 1 
        '''
        self.rcnn.eval()
        img = img.clone().detach()
        loss = 0

        if self.training:
            for (y, x), p, (h, w) in zip(self.yxs, self.patches, self.hws):
                # add patch with perturbations ranging from -0.01 to 0.01 for robustness
                img[0, :, y - h // 2: y - h // 2 + h, x - w // 2: x - w // 2 + w] += (p.type_as(img) + torch.empty_like(p).uniform_(-0.01, 0.01).type_as(img))
        else:
            for (y, x), p, (h, w) in zip(self.yxs, self.patches, self.hws):
                img[0, :, y - h // 2: y - h // 2 + h, x - w // 2: x - w // 2 + w] += p.type_as(img)

        ### punish pixels of patch which make image out of its range
        p_loss = (-img[img < 0]).sum()
        p_loss += (img[img > 1] - 1).sum()

        img = img.clamp(0, 1)
        if not self.training:
            return img

        ### faster-rcnn data stream pipeline
        img_transform = torch.nn.functional.interpolate(img * 255, (800, 800), mode='bilinear')
        img_mean = self.rcnn.cfg.data.test.pipeline[1]['transforms'][2]['mean']
        img_std = self.rcnn.cfg.data.test.pipeline[1]['transforms'][2]['std']
        for i in range(3):
            img_transform[:, i, :, :] = (img_transform[:, i, :, :] - img_mean[i]) / img_std[i]

        p_data = data.copy()
        p_data['img'] = [img_transform]

        result = self.rcnn(return_loss=False, rescale=True, **p_data)


        prob = result[0][2][:, :-1] # 1000 proposal probability of categories with shape (1000, 80)
        prob, prob_idx = prob.max(dim=-1) 
        rcnn_thres = 0.25 # t_rcnn^*

        b_prob = result[0][2][:, -1][prob > rcnn_thres] # probability whether objects are background
        r_loss = -torch.log(b_prob + 1e-3).sum()  # additional loss of background

        prob = prob[prob > rcnn_thres] # P_rcnn
        r_cnt = len(prob)
        r_loss += -(torch.clamp((prob - rcnn_thres) / (0.3 - rcnn_thres), 0, 1) * torch.log(1 - prob + 1e-3)).sum() # main faster-rcnn loss
        rcnn_loss = r_loss
        
        loss += rcnn_loss * self.ratio

        ########################################

        self.yolo.eval()

        # yolo image transform pipeline
        img_transform = torch.nn.functional.interpolate(img, (608, 608), mode='bilinear')

        ### offical data processing  
        list_boxes, _ = self.yolo(img_transform)
        anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
        num_anchors = 9
        anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        strides = [8, 16, 32]
        anchor_step = len(anchors) // num_anchors
        num_classes = 80
        b_loss = 0
        b_nms_loss = 0
        b_cnt = 0
        b_nms_cnt = 0
        yolo_thres = 0.45 # t_yolo^*
        
        for i in range(3):
            all_boxes = []
            all_boxes_nms = []
            masked_anchors = []
            for m in anchor_masks[i]:
                masked_anchors += anchors[m * anchor_step:(m + 1) * anchor_step]
            masked_anchors = [anchor / strides[i] for anchor in masked_anchors]
            
            boxes = get_region_boxes(list_boxes[i], yolo_thres, 80, masked_anchors, len(anchor_masks[i]))[0]
            
                       
            for box in boxes:
                b_cnt += 1
                b_loss += -torch.clamp((box[5] - yolo_thres) / (0.5 - yolo_thres), 0, 1) * torch.log(1 - box[5] + 1e-2) # additional loss of categories
                
                b_loss += -torch.clamp((box[4] - yolo_thres) /  (0.5 - yolo_thres), 0, 1) * torch.log(1 - box[4] + 1e-2) # main yolo loss
            
            boxes_nms = nms(boxes, 0.4)
            for box in boxes_nms:
                b_nms_cnt += 1
                b_nms_loss += -torch.clamp((box[5] - yolo_thres) /  (0.5 - yolo_thres), 0, 1) * torch.log(1 - box[5] + 1e-2) # additional loss of object probabilities of categories after nms
                    
                b_nms_loss += -torch.clamp((box[4] - yolo_thres) /  (0.5 - yolo_thres), 0, 1) * torch.log(1 - box[4] + 1e-2) # additional loss of object probabilities after nms
        
        if not isinstance(b_loss, int):
            yolo_loss = b_loss.cuda() + b_nms_loss.cuda() * (b_cnt / b_nms_cnt)
            loss += yolo_loss

        if loss < 1e-6:
            return 0
        loss += p_loss
            
        return loss


def generate_patch_parameters(rcnn_boxes, yolo_boxes, ratio, n_lines):
    '''
    :rcnn_boxes: bounding boxes detected by faster-rcnn
    :yolo_boxes: bounding boxes detected by yolo
    :ratio: the net-like patch size to minimum bounding box covering all bounding boxes
    :n_lines: n_h and n_v
    :return: a list of (y, x) of lines and a list of (h, w) of lines, these lines are parts of patch
    '''
    max_x = 0
    min_x = 500
    max_y = 0
    min_y = 500
    for (x1, y1, x2, y2, *_) in rcnn_boxes:
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        max_x = max(x2, max_x)
        min_x = min(x1, min_x)
        max_y = max(y2, max_y)
        min_y = min(y1, min_y)
    width = 500
    height = 500
    for x, y, w, h, *_ in yolo_boxes:
        x1 = max(int((x - w / 2.0) * width), 0)
        y1 = max(int((y - h / 2.0) * height), 0)
        x2 = min(int((x + w / 2.0) * width), width - 1)
        y2 = min(int((y + h / 2.0) * height), height - 1)
        max_x = max(x2, max_x)
        min_x = min(x1, min_x)
        max_y = max(y2, max_y)
        min_y = min(y1, min_y)
    yxs = []
    hws = []

    x_range = int((max_x - min_x) * ratio // 2)
    y_range = int((max_y - min_y) * ratio // 2)
    x_center = int(max_x + min_x) // 2
    y_center = int(max_y + min_y) // 2
    yxs = list(zip(np.linspace(y_center - y_range, y_center + y_range, n_lines).astype('int32'), [x_center] * n_lines)) # 贴图中心位置
    yxs += list(zip([y_center] * n_lines, np.linspace(x_center - x_range, x_center + x_range, n_lines).astype('int32')))
    hws = [[1, x_range * 2]] * n_lines + [[y_range * 2, 1]] * n_lines # 贴图尺寸
    return yxs, hws


ratio = 0.8
n_lines = 4

save_path = f't{n_lines}_{ratio:.2f}/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

if os.path.exists('checkpoint'):
    with open('checkpoint', 'rb') as f:
        last = int(f.read()) + 1
else:
    last = 0

lst = os.listdir('eval_code/select1000_new/images/')
lst.sort()
for cnt, pic in enumerate(lst[last:], last):
    print(cnt)
    img_file = 'eval_code/select1000_new/images/' + pic

    ### offical faster-rcnn pipeline
    cfg = rcnn.cfg
    device = next(rcnn.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img_file)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(rcnn.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in rcnn.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data


    with torch.no_grad():
        result = rcnn(return_loss=False, rescale=True, **data) 

    score = np.vstack([i.detach().numpy() for i in result[0][0]])
    rcnn_cnt = len(score[score[:, -1] >= 0.3])


    img = cv2.imread(img_file)
    a = cv2.resize(img, (608, 608))
    with torch.no_grad():
        yolo_boxes = do_detect(yolo, a[:, :, ::-1].copy(), 0.5, 0.4, True, False)
    yolo_cnt = len(yolo_boxes)

    img = img[:, :, ::-1].copy().transpose([2, 0, 1])[None]
    img = torch.tensor(img / 255).cuda().type(torch.float32)


    rcnn_boxes = np.vstack([i.detach().numpy() for i in result[0][0]])
    rcnn_boxes = rcnn_boxes[rcnn_boxes[:, -1] > 0.3]

    yxs, hws = generate_patch_parameters(rcnn_boxes, yolo_boxes, ratio, n_lines)

    patch_model = AdversarialPatch(yolo, rcnn, yxs, hws, 1)
    patch_model.cuda()
    opt = torch.optim.Adam(patch_model.patches, lr=0.1)

    ### adversarial training
    patch_model.train()
    for step in range(100):
        loss = patch_model(data, img)
        if isinstance(loss, int):
            break
        loss.backward()
        opt.step()
        opt.zero_grad()

    ### choose the best adversarial patch
    score_max = 0
    for step in range(10):
        patch_model.train()
        loss = patch_model(data, img)
        if not isinstance(loss, int):
            loss.backward()
            opt.step()
            opt.zero_grad()

        patch_model.eval()
        with torch.no_grad():
            img_patched = patch_model(data, img)
        p = img_patched.detach().cpu().numpy()[0].transpose([1, 2, 0])

        a = cv2.resize(p * 255, (608, 608))
        with torch.no_grad():
            j, boxes0 = do_detect(yolo, a, 0.5, 0.4, True, True)

        yolo_cnt_1 = len(boxes0)

        p_data = data.copy()
        img_transform = torch.nn.functional.interpolate(img_patched * 255, (800, 800), mode='bilinear')
        img_mean = rcnn.cfg.data.test.pipeline[1]['transforms'][2]['mean']
        img_std = rcnn.cfg.data.test.pipeline[1]['transforms'][2]['std']
        for i in range(3):
            img_transform[:, i, :, :] = (img_transform[:, i, :, :] - img_mean[i]) / img_std[i]

        p_data['img'] = [img_transform]
        with torch.no_grad():
            result = rcnn(return_loss=False, rescale=True, **p_data) 

        rcnn_boxes = np.vstack([i.detach().numpy() for i in result[0][0]])
        rcnn_cnt_1 = len(rcnn_boxes[rcnn_boxes[:, -1] >= 0.3])

        score = max(yolo_cnt - yolo_cnt_1, 0) / max(yolo_cnt, 1) + max(rcnn_cnt - rcnn_cnt_1, 0) / max(rcnn_cnt, 1)

        if score >= score_max:
            score_max = score
            img_patched_max = img_patched
        
        if score_max == 2:
            break

    ### save adversarial image
    img_patched = img_patched_max.detach().cpu().numpy()[0].transpose([1, 2, 0])
    img_patched = img_patched[:,:,::-1] * 255
    img_patched.clip(0, 255)
    img_patched = img_patched.copy().astype('uint8')
    cv2.imwrite(save_path + pic, img_patched)

    with open('checkpoint', 'w') as m:
        m.write(f'{cnt}')
    
