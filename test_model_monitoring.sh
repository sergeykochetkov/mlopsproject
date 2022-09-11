#!/bin/bash



pid=$(netstat -tulpn | grep 9696 | grep -oP '([0-9]*)/python' | grep -o '[0-9]*') && kill -9 $pid


WHYLABS_DEFAULT_ORG_ID=org-KMNvQ9
WHYLABS_API_KEY=NDWrweKLrX.uevO0DdtxMCsmcWankKzrgT8tysgpHA2CN66Jzl5TSDV0qq2P1Cqh

export WHYLABS_API_KEY=$WHYLABS_API_KEY
export WHYLABS_DEFAULT_ORG_ID=$WHYLABS_DEFAULT_ORG_ID
export WHYLABS_MODEL_ID=model-3

#run prediction service

export PYTHONPATH=$PWD:$PYTHONPATH

bash -c "python cloud_function/predict_service.py" &

sleep 2

python client_service.py

#run monitoring service

#run 1m quotes loading and sending to service

