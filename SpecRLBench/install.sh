#!/bin/bash
# Install main package
pip install -e .

# Install panda-gym submodule
pip install -e specbench/envs/panda-gym

# Install safety-gymnasium submodule
pip install -e specbench/envs/zones/safety-gymnasium
