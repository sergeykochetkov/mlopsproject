#!/bin/bash

set -x

ZIP_NAME=lambda_function.py.zip
BUCKET_NAME=serverless-functions
S3_PATH=s3://$BUCKET_NAME
ZIP_NAME_S3=$S3_PATH/$ZIP_NAME

FUNC_NAME=python-function

if true
then

yc serverless function create --name=$FUNC_NAME

if [[ -f $ZIP_NAME ]]
then
  rm $ZIP_NAME
fi

zip -r $ZIP_NAME consts.py lambda_function.py requirements.txt mlruns

fi

aws --endpoint-url=https://storage.yandexcloud.net/ s3 cp $ZIP_NAME $S3_PATH

yc serverless function version create \
  --function-name=$FUNC_NAME \
  --runtime python37 \
  --entrypoint lambda_function.handler \
  --memory 4GB \
  --execution-timeout 3s \
  --package-bucket-name $BUCKET_NAME \
  --package-object-name $ZIP_NAME