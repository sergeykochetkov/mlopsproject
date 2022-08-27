'''
flask application prediction server
'''
import mlflow
from mlflow.client import MlflowClient
import pandas as pd
from flask import Flask, request, jsonify

MLFLOW_TRACKIG_URI = 'http://127.0.0.1:5000'
RUN_ID = '93f7c132c0244a06ad08bbaf13f1332e'
logged_model = f'runs:/{RUN_ID}/model'
PORT = 9696
HOST = 'localhost'
SERVER = '/predict'
URL = f'http://{HOST}:{PORT}{SERVER}'

mlflow.set_tracking_uri(MLFLOW_TRACKIG_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKIG_URI)
path = client.download_artifacts(run_id=RUN_ID, path='model')
print(f'artifacts downloaded to {path}')

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


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
