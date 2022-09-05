# Project forecasts profitable trading position on stock market with help of pytorch and transformers.

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

To run integration test

    make run_integration_test



## What have been done:
+ Select a dataset that you're interested in


    I use stock prices data from yahoo.finance, and forecast the most profitable position for this stock


+ Train a model on that dataset tracking your experiments


    mlflow is used for experiment tracking

+ Create a model training pipeline

    
    prefect is used for training pipeline, and deploying training pipeline


+ Deploy the model in batch, web service or streaming

    
    model is (almost) deployed as streaming for yandex.cloud cloud function (analog for AWS lambda).
    Lambda function is written and tested locally (test_lambda_function_local.py).
    yandex.cloud cloud function is created by script create_cloud_function.sh via zipping of code and model and s3
    message queues are created and tested (analogs for aws kinesis)


    model is containerazed in Docker (see lambda/run_integration_test.sh)


+ Monitor the performance of your model


    not done


+ Follow the best practices


    unittests are implemented. See test_*.py
    
    git pre-commit is used (see script pre-commit)
    
    make is used for environment setup, unit testing, linting
    
    pylint is used

    there is an integration test lambda/run_integration_test.sh
