#!/usr/bin/env python

import sys
import rospkg
import rospy
import roslib
from vlm.msg import StampedString

def main(): 
    instruction_pub = rospy.Publisher("/vlm/instruction", StampedString, queue_size=10)

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():

        msg = StampedString()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "instruction"
        msg.data = """
        Respond with only one of these phrases describing the quadrant of the image that the banana is in:
        - top left
        - top right
        - bottom left
        - bottom right
        """

        instruction_pub.publish(msg)

        rate.sleep()

    return 0

if __name__ == "__main__":
    rospy.init_node("instruction")
    sys.exit(main())