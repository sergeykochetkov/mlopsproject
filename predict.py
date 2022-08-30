'''
flask application prediction server
'''
import os

import mlflow
import pandas as pd
from flask import Flask, request, jsonify

MLFLOW_TRACKIG_URI = 'http://127.0.0.1:5000'
RUN_ID = '93f7c132c0244a06ad08bbaf13f1332e'

PORT = 9696
HOST = '127.0.0.1'
SERVER = '/predict'
URL = f'http://{HOST}:{PORT}{SERVER}'


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


def predict_on_df(data: pd.DataFrame):
    '''
    calls model on input dataframe
    '''
    x_features = pd.DataFrame([data])
    return loaded_model.predict(x_features)


app = Flask('stock-prediction')


@app.route(SERVER, methods=['POST'])
def predict_endpoint():
    '''
    gets sent json data, calls model, returns output json back
    '''
    stock_data = request.get_json()
    feature_x = stock_data['x']
    pred = predict_on_df(feature_x)
    pred = pred.iloc[0, 0]
    result = {
        'prediction': float(pred),
        'time': stock_data['time'],
        'y': stock_data['y']
    }

    return jsonify(result)


def main():
    '''
    start Flask app, call this in separate process
    '''
    app.run(debug=True, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
