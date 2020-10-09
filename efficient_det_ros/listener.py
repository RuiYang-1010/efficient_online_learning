#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from vision_msgs.msg import BoundingBox2D, ObjectHypothesisWithPose, Detection2D, Detection2DArray

def callback(data):
    rospy.loginfo(data)
    #rospy.loginfo(data.header.seq)
    #rospy.loginfo(data.header.stamp)

def listener():

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/image_detections", Detection2DArray, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
