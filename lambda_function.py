'''
methods and variables for serving model via lambda-function on aws
 or cloud-functions on yandex.cloud
'''

import os
import traceback
import io

import mlflow
import pandas as pd
from consts import RUN_ID


def get_model_path():
    '''
    if environmental variable MODEL_LOCATION is set - returns it
    otherwise seeks model in mlflow model tracking db by RUN_ID.
    Needed for integration_test, where we do not use mlflow model registry,
     but load model from local path
    '''

    model_location = os.getenv('MODEL_LOCATION')
    if model_location:
        return model_location

    logged_model = f'runs:/{RUN_ID}/model'
    return logged_model


# Load model as a PyFuncModel.
model_path = get_model_path()
print(f' load model from {model_path}')

loaded_model = mlflow.pyfunc.load_model(model_path)


def predict_on_df(data: pd.DataFrame) -> float:
    '''
    calls model on input dataframe
    '''
    x_features = pd.DataFrame([data])
    output = loaded_model.predict(x_features)
    output = float(output.iloc[0, 0])
    return output


def handler(event, context):
    '''
    lambda function handler for serverless function
    :param event: dict of inputs
    :param context: additional data, not used yet
    :return: dict with predictions of model and error handling message
    '''
    assert context is None

    print(event)

    request_id = -1
    try:
        request_id = event['request_id']
        features_x = event['features_x']

        prediction = predict_on_df(features_x)
        status_code = 0
        error_msg = ''
    except KeyError as exception:
        status_code = -2
        prediction = -1
        error_msg = str(exception)
    except RuntimeError:
        with io.StringIO() as output:
            traceback.print_exc(file=output)
            error_msg = output.getvalue()
        status_code = -3
        prediction = -1
    except Exception as exception:  # pylint: disable=broad-except
        status_code = -1
        prediction = -1
        error_msg = str(exception)
    return {
        'request_id': request_id,
        'status_code': status_code,
        'prediction': prediction,
        'error_msg': error_msg
    }
