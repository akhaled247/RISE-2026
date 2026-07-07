#!/usr/bin/env python

import sys
import os
import numpy as np
import rospkg
import rospy
import base64
import roslib
import message_filters
import cv2
from vlm.msg import StampedString
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from openai import OpenAI

class VLM:
    def __init__(self, base_url, model):
        self.bridge = CvBridge()
        instruction_sub = message_filters.Subscriber("/vlm/instruction", StampedString)
        image_sub = message_filters.Subscriber("/io/internal_camera/head_camera/image_raw", Image)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [instruction_sub, image_sub],
            queue_size=1,
            slop=0.1
        )

        self.sync.registerCallback(self.callback)

        self.image = None
        self.instruction = None
        self.client = OpenAI(
            api_key="dummy",
            base_url=base_url
        )
        self.model = model
  
    def callback(self, instruction_msg, image_msg):
        self.instruction = instruction_msg.data

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        success, buffer = cv2.imencode(".jpg", cv_image)

        if not success:
            rospy.logerr("Failed to encode image")
            return

        self.image = base64.b64encode(buffer).decode("utf-8")

    def get_response(self, message, base64_image):
        response = self.client.chat.completions.create(
            model=self.model,
            #reasoning_effort="low",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    
    def get_current_prompt(self):
        return self.instruction, self.image

  
def main(): 
    vlm = VLM(base_url="http://localhost:49173/v1", model="Qwen/Qwen3.5-4B")
    
    rate = rospy.Rate(0.1)
    while not rospy.is_shutdown():
        print("Fetching response...")

        start_time = rospy.Time.now()

        instruction, image = vlm.get_current_prompt()

        if image is not None and instruction is not None:
            try:
                response = vlm.get_response(instruction, image)
                print("Response:")
                print(response)
            except Exception as e:
                rospy.logerr(f"VLM request failed: {e}")
        else:
            if image is None:
                print("image is None")
            if instruction is None:
                print("instruction is None")

        end_time = rospy.Time.now()
        duration_s = (end_time - start_time).to_sec()
        print(f"Inference took {duration_s:.3f} seconds")

        rate.sleep()

    return 0

if __name__ == "__main__":
    rospy.init_node("vlm_node")
    sys.exit(main())