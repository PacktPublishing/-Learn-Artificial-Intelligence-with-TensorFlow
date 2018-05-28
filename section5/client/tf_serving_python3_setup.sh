#!/bin/bash

# Download pip package.
wget https://pypi.python.org/packages/37/82/cbbf1f2aef8e6e73fa26c0c3a88ace022c774e8b5d0e10ca7a58fd3cdee4/tensorflow_serving_api-1.4.0-py2-none-any.whl#md5=efd9cce40839a6ff122d3971e83925c4

# Extract it.
unzip tensorflow_serving_api-1.4.0-py2-none-any.whl

# Copy it to python3.5 libs.
sudo cp -r tensorflow_serving tensorflow_serving_api-1.4.0.dist-info /usr/local/lib/python3.5/dist-packages/

# If you get permission denied when trying to import in python:
sudo chown -R "$(whoami):$(whoami)" /usr/local/lib/python3.5/dist-packages/tensorflow_serving*



