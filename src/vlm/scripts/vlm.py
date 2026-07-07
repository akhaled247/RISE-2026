#!/usr/bin/env python

import numpy as np
import rospy
from openai import OpenAI

def main(): 
    client = OpenAI(
        api_key="dummy",
        base_url="http://localhost:49173/v1"
    )

    response = client.responses.create(
        model="Qwen/Qwen3.5-4B",
        reasoning={"effort": "none"},
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Who are you?"},
            ],
        }],
    )

    print("RESPONSE:", response.output_text)

if __name__ == "__main__":
    rospy.init_node("vlm")
    main()
    rospy.spin()