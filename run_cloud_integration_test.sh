#!/bin/bash

set -x

# URL got during creating yandex.cloud serverless container
YC_URL="https://bbaaeof3gfmonespgi81.containers.yandexcloud.net/predict"
YC_TOKEN=$( yc iam create-token )

export PYTHONPATH=$PWD:$PYTHONPATH

python client_service.py --yc_token="$YC_TOKEN" --yc_container_url="$YC_URL"

