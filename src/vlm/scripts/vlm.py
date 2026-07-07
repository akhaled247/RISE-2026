#!/usr/bin/env python

import sys
import rospkg
import numpy as np
import rospy
import base64
import os
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main(): 
    client = OpenAI(
        api_key="dummy",
        base_url="http://localhost:49173/v1"
    )

    rospack = rospkg.RosPack()
    image_path = os.path.join(rospack.get_path('vlm'), 'assets', 'leclerc.jpg')
    base64_image = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3.5-4B",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Who is in this image?"},
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
    except Exception as e:
        print("Error", e)
        sys.exit(1)

    print("RESPONSE:")
    print(response.choices[0].message.content)
    sys.exit(0)

if __name__ == "__main__":
    rospy.init_node("vlm")
    main()
    rospy.spin()