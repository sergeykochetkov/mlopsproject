# Project forecasts profitable trading position on stock market with help of pytorch and transformers. Graduation Project for MLOps course https://github.com/DataTalksClub/mlops-zoomcamp

## We use stock ticker, history length, time intervals and other parameters specified in main.py. 

## Overview

To install environment run

    make setup

For unittests run

    make test


To train and validate pytorch transformer run

    python main.py

To deploy training pipeline run

    make prefect_deploy_main

To run integration test locally

    make run_integration_test  OR ./run_local_integration_test.sh

To run integration test on cloud

    ./run_cloud_integration_test.sh


## What have been done:
+ Select a dataset that you're interested in


    I use stock prices data from yahoo.finance, and forecast the most profitable position for this stock


+ Train a model on that dataset tracking your experiments


    mlflow is used for experiment tracking

+ Create a model training pipeline

    
    prefect is used for training pipeline, and deploying training pipeline


+ Deploy the model in batch, web service or streaming

    
    model is deployed as streaming for yandex.cloud serverless container (analog for AWS lambda with container).
    Lambda function is written and tested locally (test_lambda_function_local.py).
    model is containerazed in Docker serverless_container.Dockerfile

    to deploy model in streaming mode to yandex.cloud serverless container do:
    1. change adress to your yandex.cloud container registry in script ./docker_push_to_yc.sh  
    2. ./docker_push_to_yc.sh [your_tag]
    3. create container revision in yandex.cloud with container pushed in 2.
    4. add environmental variable to this revision MODEL_LOCATION=/app/cloud_function/mlruns/0/d8c76e5a2d3a4419a8305a361d351b36/artifacts/model
    5. on your local machine run client server sending requests to cloud by command ./run_cloud_integration_test.sh
    important: yandex.cloud works only with PORT=8080 defined in cloud_function/consts.py


+ Monitor the performance of your model


    not done


+ Follow the best practices


    unittests are implemented. See test_*.py
    
    git pre-commit is used (see script pre-commit)
    
    make is used for environment setup, unit testing, linting
    
    pylint is used

    there is an integration test lambda/run_integration_test.sh
