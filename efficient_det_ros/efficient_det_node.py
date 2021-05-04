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

obj_list = ['car', 'person', 'cyclist']

compound_coef = 2
force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
#anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
#anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
anchor_ratios = [(0.6, 1.5), (1.1, 0.9), (1.5, 0.7)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

color_list = standard_to_bgr(STANDARD_COLORS)

img_path = '/home/epan/Rui/datasets/2011_09_26_drive_0005_sync/img'
txt_path = '/home/epan/Rui/datasets/2011_09_26_drive_0005_sync/txt'

path = '/home/epan/Rui/datasets/2011_09_26_drive_0005_sync/image_02/data'
stamp_path = '/home/epan/Rui/datasets/2011_09_26_drive_0005_sync/image_02/timestamps.txt'


def display(cur_frame, preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])

        if imshow:
            cv2.imshow('img', imgs[i])
            #cv2.waitKey(0)

        if imwrite:
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            cv2.imwrite(
                f'{img_path}/img_inferred_d{compound_coef}_this_repo_{cur_frame}.jpg', imgs[i])


def image_callback():
    rospy.loginfoV("Get an image")


def EfficientDetNode():
    rospy.init_node('efficient_det_node', anonymous=True)
    rospy.Subscriber('input', String, image_callback, queue_size=1)
    pub = rospy.Publisher('/image_detections', Detection2DArray, queue_size=10)
    rate = rospy.Rate(1)  # 10hz

    path_list = os.listdir(path)
    path_list.sort(key=lambda x: int(x.split('.')[0]))

    stamp_file = open(stamp_path)
    stamp_lines = stamp_file.readlines()
    stamp_i = 0

    for filename in path_list:
        img_path = filename
        cur_frame = img_path[:-4]
        img_path = path + "/" + img_path

        cur_stamp = ((float)(stamp_lines[stamp_i][-13:].strip('\n')))
        # cur_stamp = rospy.Time.from_sec(
        #     ((float)(stamp_lines[stamp_i][-13:].strip('\n'))))
        stamp_i += 1

        detection_results = Detection2DArray()

        # tf bilinear interpolation is different from any other's, just make do
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
        ori_imgs, framed_imgs, framed_metas = preprocess(
            img_path, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda()
                             for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(
            0, 3, 1, 2)

        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=anchor_ratios, scales=anchor_scales)
        model.load_state_dict(torch.load(
            f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
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

        display(cur_frame, out, ori_imgs, imshow=False, imwrite=True)

        for i in range(len(out)):
            for j in range(len(out[i]['rois'])):
                x1, y1, x2, y2 = out[i]['rois'][j].astype(np.int)
                obj = obj_list[out[i]['class_ids'][j]]
                score = float(out[i]['scores'][j])

                result = ObjectHypothesisWithPose()
                result.score = score
                if(obj == 'car'):
                    result.id = 0
                if(obj == 'person'):
                    result.id = 1
                if(obj == 'cyclist'):
                    result.id = 2

                detection_msg = Detection2D()
                detection_msg.bbox.center.x = (x1 + x2) / 2
                detection_msg.bbox.center.y = (y1 + y2) / 2
                detection_msg.bbox.size_x = x2 - x1
                detection_msg.bbox.size_y = y2 - y1

                detection_msg.results.append(result)
                detection_results.detections.append(detection_msg)
                rospy.loginfo("%d: %lf", detection_msg.results[0].id, detection_msg.results[0].score)

            detection_results.header.seq = cur_frame
            #detection_results.header.stamp = cur_stamp
            rospy.loginfo(detection_results.header.stamp)
            pub.publish(detection_results)

            if not os.path.exists(txt_path):
                os.makedirs(txt_path)
            #with open(f'txt/{cur_frame}.txt', 'w') as f:
            with open(f'{txt_path}/{cur_frame}.txt', 'w') as f:
                #f.write(str((float)(stamp_lines[stamp_i][-13:].strip('\n'))) + "\n")
                f.write(str(cur_stamp) + "\n")
                for detection in detection_results.detections:
                    f.write(str(detection.bbox.center.x) + " ")
                    f.write(str(detection.bbox.center.y) + " ")
                    f.write(str(detection.bbox.size_x) + " ")
                    f.write(str(detection.bbox.size_y) + " ")
                    f.write(str(detection.results[0].id) + " ")
                    f.write(str(detection.results[0].score) + "\n")
            f.close()

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
