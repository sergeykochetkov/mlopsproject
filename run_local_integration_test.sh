#!/bin/bash

set -x

pid=$(netstat -tulpn | grep 9696 | grep -oP '([0-9]*)/python' | grep -o '[0-9]*') && kill -9 $pid && echo $pid

ENV_NAME=MLOpsProject

LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
export LOCAL_IMAGE_NAME="predict_service:${LOCAL_TAG}"
echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
docker build -f serverless_container.Dockerfile -t ${LOCAL_IMAGE_NAME} . &&

container_id=$(docker run -p 9696:9696 --env MODEL_LOCATION=/app/cloud_function/mlruns/0/d8c76e5a2d3a4419a8305a361d351b36/artifacts/model -d -t ${LOCAL_IMAGE_NAME}) &&

echo "run container ${container_id}"

sleep 5

CA="source /home/$USER/anaconda3/etc/profile.d/conda.sh && activate ${ENV_NAME}"

export PYTHONPATH="${PWD}/../:$PYTHONPATH" && $CA && python integration_test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ];
then
  docker logs $container_id
  docker stop $container_id
  exit ${ERROR_CODE}
fi

docker stop $container_id
