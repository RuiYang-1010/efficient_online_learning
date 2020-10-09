#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from vision_msgs.msg import BoundingBox2D, ObjectHypothesisWithPose, Detection2D, Detection2DArray


import time
import torch
import cv2
import numpy as np
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video

# Video's path
video_src = 'test/video_kitti.mp4'  # set int to use webcam, set str to read from a video file

compound_coef = 0
force_input_size = None  # set None to use default size

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

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

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

# load model
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

# function for display
def display(preds, imgs):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

        return imgs[i]
# Box
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

# Video capture
cap = cv2.VideoCapture(video_src)
cur_frame_id = 0;

def video_callback():
    rospy.loginfoV("Got an video")

def EfficientDetNode():
    rospy.Subscriber('input', String, video_callback, queue_size=1)
    pub = rospy.Publisher('/image_detections', Detection2DArray, queue_size=10)
    rate = rospy.Rate(10) # 10hz

    detection_results = Detection2DArray();

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame preprocessing
        global cur_frame_id
        cur_frame_id += 1
        ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        # model predict
        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

        # result
        out = invert_affine(framed_metas, out)
        img_show = display(out, ori_imgs)

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
                    rospy.loginfo(detection_msg.results[0].id)

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
                    rospy.loginfo(detection_msg.results[0].id)

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
                    rospy.loginfo(detection_msg.results[0].id)

            detection_results.header.frame_id = str(cur_frame_id)
            pub.publish(detection_results)
            rate.sleep()

        # show frame by frame
        cv2.imshow('frame',img_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('efficient_det_node', anonymous=True)
    try:
        EfficientDetNode()
    except rospy.ROSInterruptException:
        pass
