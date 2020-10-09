#!/usr/bin/env python

import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

import os
import rospy
from std_msgs.msg import String
from vision_msgs.msg import BoundingBox2D, ObjectHypothesisWithPose, Detection2D, Detection2DArray

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

color_list = standard_to_bgr(STANDARD_COLORS)

def display(preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])

def image_callback():
    rospy.loginfoV("Get an image")

def EfficientDetNode():
    rospy.init_node('efficient_det_node', anonymous=True)
    rospy.Subscriber('input', String, image_callback, queue_size=1)
    pub = rospy.Publisher('/image_detections', Detection2DArray, queue_size=10)
    rate = rospy.Rate(10) # 10hz

    path = '/home/rui/Rui/datasets/KITTI_Odometry/00/image'
    stamp_path = '/home/rui/Rui/datasets/KITTI_Odometry/00/times.txt'
    path_list = os.listdir(path)
    path_list.sort(key=lambda x:int(x.split('.')[0]))

    stamp_file = open(stamp_path)
    stamp_lines = stamp_file.readlines()
    stamp_i = 0

    for filename in path_list:
        img_path = filename
        cur_frame = img_path[:-4]
        img_path = path + "/" + img_path

        cur_stamp = ((float)(stamp_lines[stamp_i].strip('\n'))).time()
        stamp_i += 1

        detection_results = Detection2DArray();

        compound_coef = 0
        force_input_size = None  # set None to use default size

        # replace this part with your project's anchor config
        anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

        threshold = 0.2
        iou_threshold = 0.2

        use_cuda = True
        use_float16 = False
        cudnn.fastest = True
        cudnn.benchmark = True

        # tf bilinear interpolation is different from any other's, just make do
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=anchor_ratios, scales=anchor_scales)
        model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model = model.cuda()
        if use_float16:
            model = model.half()

        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)

        out = invert_affine(framed_metas, out)
        display(out, ori_imgs, imshow=False, imwrite=False)

        for i in range(len(out)):
            for j in range(len(out[i]['rois'])):
                x1, y1, x2, y2 = out[i]['rois'][j].astype(np.int)
                obj = obj_list[out[i]['class_ids'][j]]
                score = float(out[i]['scores'][j])

                if(obj == 'car' or obj == 'bus' or obj == 'truck'):
                    detection_msg = Detection2D()
                    detection_msg.bbox.center.x = (x1+x2)/2
                    detection_msg.bbox.center.y = (y1+y2)/2
                    detection_msg.bbox.size_x = x2-x1
                    detection_msg.bbox.size_y = y2-y1
                    result = ObjectHypothesisWithPose()
                    result.id = 0
                    result.score = score
                    detection_msg.results.append(result)
                    detection_results.detections.append(detection_msg)
                    rospy.loginfo("%d:%lf",detection_msg.results[0].id,detection_msg.results[0].score)

                if(obj == 'person'):
                    detection_msg = Detection2D()
                    detection_msg.bbox.center.x = (x1+x2)/2
                    detection_msg.bbox.center.y = (y1+y2)/2
                    detection_msg.bbox.size_x = x2-x1
                    detection_msg.bbox.size_y = y2-y1
                    result = ObjectHypothesisWithPose()
                    result.id = 1
                    result.score = score
                    detection_msg.results.append(result)
                    detection_results.detections.append(detection_msg)
                    rospy.loginfo("%d:%lf",detection_msg.results[0].id,detection_msg.results[0].score)

                if(obj == 'bicycle' or obj == 'motorcycle'):
                    detection_msg = Detection2D()
                    detection_msg.bbox.center.x = (x1+x2)/2
                    detection_msg.bbox.center.y = (y1+y2)/2
                    detection_msg.bbox.size_x = x2-x1
                    detection_msg.bbox.size_y = y2-y1
                    result = ObjectHypothesisWithPose()
                    result.id = 2
                    result.score = score
                    detection_msg.results.append(result)
                    detection_results.detections.append(detection_msg)
                    rospy.loginfo("%d:%lf",detection_msg.results[0].id,detection_msg.results[0].score)

            detection_results.header.seq = cur_frame
            detection_results.header.stamp = cur_stamp
            rospy.loginfo(detection_results)
            pub.publish(detection_results)
            rate.sleep()

        print('running speed test...')
        with torch.no_grad():
            print('test1: model inferring and postprocessing')
            print('inferring image for 10 times...')
            t1 = time.time()
            for _ in range(10):
                _, regression, classification, anchors = model(x)

                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  threshold, iou_threshold)
                out = invert_affine(framed_metas, out)

            t2 = time.time()
            tact_time = (t2 - t1) / 10
            print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

if __name__ == '__main__':
    try:
        EfficientDetNode()
    except rospy.ROSInterruptException:
        pass
