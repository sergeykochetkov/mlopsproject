'''
sanic application prediction server
'''
import os
from sanic import Sanic, json
from cloud_function.consts import SERVER, HOST, PORT

from cloud_function.lambda_function import predict_on_df

app = Sanic('stock-prediction')


@app.route(SERVER)
async def predict_endpoint(request):
    '''
    gets sent json data, calls model, returns output json back
    '''

    stock_data = request.get_json()
    feature_x = stock_data['x']
    pred = predict_on_df(feature_x)
    result = {
        'prediction': pred,
        'time': stock_data['time'],
        'y': stock_data['y']
    }

    return json(result)


def main():
    '''
    start Sanic app, call this in separate process
    '''
    if 'PORT' in os.environ:
        port = os.environ['PORT']
        host = '0.0.0.0'
    else:
        port = PORT
        host = HOST
    app.run(host=host, port=port, motd=False, access_log=False, debug=True)


if __name__ == "__main__":
    main()
