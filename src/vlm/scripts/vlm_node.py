#!/usr/bin/env python

import sys
import base64
import rospy
import message_filters
import cv2

from vlm.msg import StampedString
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from openai import OpenAI


class VLM:
    def __init__(self, base_url, model):
        self.bridge = CvBridge()

        self.instruction_sub = message_filters.Subscriber(
            "/vlm/instruction", StampedString
        )
        self.image_sub = message_filters.Subscriber(
            "/io/internal_camera/head_camera/image_raw", Image
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.instruction_sub, self.image_sub], queue_size=5, slop=0.1
        )
        self.sync.registerCallback(self.callback)

        self.used_image_pub = rospy.Publisher(
            "/vlm/used/image_raw", Image, queue_size=1, latch=True
        )
        self.used_instruction_pub = rospy.Publisher(
            "/vlm/used/instruction", StampedString, queue_size=1, latch=True
        )

        self.output_pub = rospy.Publisher("/vlm/output", StampedString, queue_size=1)

        self.latest_pair = None

        self.client = OpenAI(api_key="dummy", base_url=base_url)
        self.model = model

    def callback(self, instruction_msg, image_msg):
        self.latest_pair = (instruction_msg, image_msg)

    def image_msg_to_base64(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image message: {e}")
            return None

        success, buffer = cv2.imencode(".jpg", cv_image)

        if not success:
            rospy.logerr("Failed to encode image")
            return None

        return base64.b64encode(buffer).decode("utf-8")

    def process_raw_output(self, raw_output):
        return raw_output.split("\n")[-1]

    def output_response(self, instruction_msg, image_msg):
        self.used_instruction_pub.publish(instruction_msg)
        self.used_image_pub.publish(image_msg)

        base64_image = self.image_msg_to_base64(image_msg)

        if base64_image is None:
            raise RuntimeError("Could not encode image to base64")

        response = self.client.chat.completions.create(
            model=self.model,
            reasoning_effort="none",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction_msg.data},
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

        raw_output = response.choices[0].message.content
        output = self.process_raw_output(raw_output)

        output_msg = StampedString()
        output_msg.header.stamp = rospy.Time.now()
        output_msg.header.frame_id = "output"
        output_msg.data = output

        self.output_pub.publish(output_msg)

        return raw_output, output

    def get_current_prompt(self):
        pair = self.latest_pair

        if pair is None:
            return None, None

        instruction_msg, image_msg = pair
        return instruction_msg, image_msg


def main():
    vlm = VLM(base_url="http://localhost:49173/v1", model="Qwen/Qwen3.5-4B")

    rate = rospy.Rate(0.5)

    while not rospy.is_shutdown():
        print("Fetching response...")

        start_time = rospy.Time.now()

        instruction_msg, image_msg = vlm.get_current_prompt()

        if image_msg is not None and instruction_msg is not None:
            try:
                raw_output, output = vlm.output_response(instruction_msg, image_msg)
                print("Response:")
                print(raw_output)
            except Exception as e:
                rospy.logerr(f"VLM request failed: {e}")
        else:
            if image_msg is None:
                print("image_msg is None")
            if instruction_msg is None:
                print("instruction_msg is None")

        end_time = rospy.Time.now()
        duration_s = (end_time - start_time).to_sec()
        print(f"Inference took {duration_s:.3f} seconds")

        rate.sleep()

    return 0


if __name__ == "__main__":
    rospy.init_node("vlm_node")
    sys.exit(main())
