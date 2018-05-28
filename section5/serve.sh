#!/bin/bash

tensorflow_model_server \
    --port=9000 \
    --model_base_path=$PWD/export_dir
