#!/usr/bin/env python

from tf.transformations import quaternion_matrix
import numpy as np
import rospy
import tf2_ros
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TransformStamped

OBJECT_NAMES = ["red_block_1", "red_block_2","red_block_3","sawyer"]

class GazeboObjectTFBroadcaster: 
    def __init__(self):
        self.br = tf2_ros.TransformBroadcaster()
        self.last_stamp = rospy.Time(0)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback)

    def callback(self, msg):
        stamp = rospy.Time.now()

        if stamp == self.last_stamp:
            return
        self.last_stamp = stamp

        for object_name in OBJECT_NAMES:
            if object_name not in msg.name:
                rospy.logwarn_throttle(2.0, "Object '%s' not found in /gazebo/model_states", object_name)
                continue

            idx = msg.name.index(object_name)
            pose = msg.pose[idx]

            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "world"
            t.child_frame_id = object_name

            t.transform.translation.x = pose.position.x
            t.transform.translation.y = pose.position.y
            t.transform.translation.z = pose.position.z

            if object_name == "block":
                p = np.array([
                    pose.position.x,
                    pose.position.y,
                    pose.position.z
                ])

                q = [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w
                ]

                R = quaternion_matrix(q)[:3, :3]

                offset_local = np.array([0.025, 0.025, 0.025])
                center = p + R.dot(offset_local)

                t.transform.translation.x = center[0]
                t.transform.translation.y = center[1]
                t.transform.translation.z = center[2]

            t.transform.rotation = pose.orientation

            self.br.sendTransform(t)

if __name__ == "__main__":
    rospy.init_node("gazebo_object_tf_broadcaster")
    GazeboObjectTFBroadcaster()
    rospy.spin()