#!/usr/bin/env python

import numpy as np
import rospy
import openai

def main(): 
    from openai import OpenAI

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
    main()
    rospy.init_node("vlm")
    rospy.spin()