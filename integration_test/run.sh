#!/usr/bin/env bash

set -x

LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
export LOCAL_IMAGE_NAME="predict_service:${LOCAL_TAG}"
echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
docker build -t ${LOCAL_IMAGE_NAME} ..

docker-compose up -d

sleep 5

CA="source ~/anaconda3/etc/profile.d/conda.sh && activate ${ENV_NAME}"

export PYTHONPATH=${PWD}/../;$PYTHONPATH

$CA && python integration_test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ];
then
  docker-compose logs
  docker-compose down
  exit ${ERROR_CODE}
fi

docker-compose down
