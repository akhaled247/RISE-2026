#!/usr/bin/env python

import sys
import rospkg
import rospy
import roslib
from vlm.msg import StampedString
from geometry_msgs.msg import PointStamped

def main(): 
    instruction_pub = rospy.Publisher("/vlm/ground_truth/table_relative_position", PointStamped, queue_size=10)

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():

        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "instruction"
        msg.data = """
        Find the quadrant of the table that the banana is in.

        Your final answer is the last line of your response. It must be one of these phrases:
        - top left
        - top right
        - bottom left
        - bottom right
        - none
        """

        instruction_pub.publish(msg)

        rate.sleep()

    return 0

if __name__ == "__main__":
    rospy.init_node("instruction")
    sys.exit(main())