#!/bin/bash

set -x

ENV_NAME=MLOpsProject

LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
export LOCAL_IMAGE_NAME="predict_service:${LOCAL_TAG}"
echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
docker build -t ${LOCAL_IMAGE_NAME} ..

container_id=$(docker run -p 9696:9696 --env MODEL_LOCATION=/app/mlruns/0/93f7c132c0244a06ad08bbaf13f1332e/artifacts/model -d -t ${LOCAL_IMAGE_NAME})

echo "run container ${container_id}"

sleep 2

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
