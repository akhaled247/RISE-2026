#!/usr/bin/env python

# Copyright (c) 2015-2018, Rethink Robotics Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sawyer SDK Inverse Kinematics Pick and Place Demo
"""
import argparse
import struct
import sys
import copy

import tf
import rospy
import rospkg

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
    Vector3
)

import intera_interface

sawyer_pos = Point()
block_pos = Point()
class PickAndPlace(object):
    def __init__(self, limb="right", hover_distance = 0.15, tip_name="right_gripper_tip", sawyer_pos=Point(), block_pos=Point()):
        self._limb_name = limb # string
        self._tip_name = tip_name # string
        self._hover_distance = hover_distance # in meters
        self._limb = intera_interface.Limb(limb)
        self._gripper = intera_interface.Gripper()
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        self._limb.set_joint_position_speed(0.0)

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        self.gripper_open()

    def _guarded_move_to_joint_position(self, joint_angles, timeout=5.0):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles,timeout=timeout)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(0.5)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(0.5)

    def _approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self._limb.ik_request(approach, self._tip_name)
        self._limb.set_joint_position_speed(0.001)
        self._guarded_move_to_joint_position(joint_angles)
        self._limb.set_joint_position_speed(0.1)

    def _retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        self._servo_to_pose(ik_pose)

    def _servo_to_pose(self, pose, time=2.0, steps=1.0):
        ''' Cartesian move '''
        rospy.sleep(0.5)
        if rospy.is_shutdown():
            return

        ik_step = Pose()
        ik_step.position.x = pose.position.x
        ik_step.position.y = pose.position.y
        ik_step.position.z = pose.position.z
        ik_step.orientation.x = pose.orientation.x
        ik_step.orientation.y = pose.orientation.y
        ik_step.orientation.z = pose.orientation.z
        ik_step.orientation.w = pose.orientation.w
        self._limb.set_joint_position_speed(0.0)
        joint_angles = self._limb.ik_request(ik_step, self._tip_name)
        if joint_angles:
            self._limb.set_joint_positions(joint_angles)
        else:
            return False
        rospy.sleep(0.5)

    def pick(self, pose):
        if rospy.is_shutdown():
            return

        approach = copy.deepcopy(pose)
        approach.position.z = approach.position.z + self._hover_distance

        self.gripper_open()

        if self._servo_to_pose(approach) is False:
            return False
        if self._servo_to_pose(pose) is False:
            return False

        rospy.sleep(0.5)

        self.gripper_close()
        if self._servo_to_pose(approach) is False:
            return False
        

    def place(self, pose):
        approach = copy.deepcopy(pose)
        approach.position.z = approach.position.z + self._hover_distance


        if self._servo_to_pose(approach) is False:
            return False
        if self._servo_to_pose(pose) is False:
            return False


        self.gripper_open()
        if self._servo_to_pose(approach) is False:
            return False

def load_gazebo_models(table_pose=Pose(position=Point(x=0.75, y=0.0, z=0.0)),
                       table_reference_frame="world",
                       block_pose=Pose(position=Point(x=0.4225, y=0.1265, z=0.7725)),
                       block_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('sawyer_sim_examples')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "cafe_table/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    # Load Block URDF
    block_xml = ''
    with open (model_path + "block/model.urdf", "r") as block_file:
        block_xml=block_file.read().replace('\n', '')
    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("cafe_table", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException as e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block", block_xml, "/",
                               block_pose, block_reference_frame)
    except rospy.ServiceException as e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))

def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("cafe_table")
        resp_delete = delete_model("block")
    except rospy.ServiceException as e:
        print("Delete Model service call failed: {0}".format(e))

def main():
    """SDK Inverse Kinematics Pick and Place Example

    A Pick and Place example using the Rethink Inverse Kinematics
    Service which returns the joint angles a requested Cartesian Pose.
    This ROS Service client is used to request both pick and place
    poses in the /base frame of the robot.

    Note: This is a highly scripted and tuned demo. The object location
    is "known" and movement is done completely open loop. It is expected
    behavior that Sawyer will eventually mis-pick or drop the block. You
    can improve on this demo by adding perception and feedback to close
    the loop.
    """
    rospy.init_node("ik_pick_and_place_demo")
    # Load Gazebo Models via Spawning Services
    # Note that the models reference is the /world frame
    # and the IK operates with respect to the /base frame
    # Remove models from the scene on shutdown
    rospy.on_shutdown(delete_gazebo_models)

    limb = 'right'
    hover_distance = 0.15 # meters
    # Starting Joint angles for right arm
    starting_joint_angles = {'right_j0': -0.041662954890248294,
                             'right_j1': -1.0258291091425074,
                             'right_j2': 0.0293680414401436,
                             'right_j3': 2.17518162913313,
                             'right_j4':  -0.06703022873354225,
                             'right_j5': 0.3968371433926965,
                             'right_j6': 1.7659649178699421}
    pnp = PickAndPlace(limb, hover_distance)
    listener = tf.TransformListener()
    # An orientation for gripper fingers to be overhead and parallel to the obj
    overhead_orientation = Quaternion(
                             x=-0.00142460053167,
                             y=0.999994209902,
                             z=-0.00177030764765,
                             w=0.00253311793936)

    # Move to the desired starting angles
    pnp.move_to_start(starting_joint_angles)

    while not rospy.is_shutdown():
        print("Picking...")
        (translation, rotation) = listener.lookupTransform('sawyer', 'banana', rospy.Time(0))
        while pnp.pick(Pose(position=Point(x=translation[0]+0.02,y=translation[1],z=translation[2]), orientation=overhead_orientation)) is False:
            print("Attempting picking again...")
            (translation, rotation) = listener.lookupTransform('sawyer', 'banana', rospy.Time(0))

        print("Placing...")
        pnp.place(Pose(position=Point(x=translation[0]+0.1,y=translation[1],z=translation[2]), orientation=overhead_orientation))
    return 0

if __name__ == '__main__':
    sys.exit(main())
	
