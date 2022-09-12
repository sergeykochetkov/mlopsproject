'''
flask application prediction server
'''
from flask import Flask, request, jsonify
from cloud_function.consts import SERVER, HOST, PORT
from cloud_function.lambda_function import predict_on_df


def main():
    '''
    start Flask app, call this in separate process
    '''
    app = Flask('stock-prediction')

    @app.route(SERVER, methods=['POST'])
    def predict_endpoint():
        '''
        gets sent json data, calls model, returns output json back
        '''
        print('inside predict service')
        stock_data = request.get_json()
        print(f'got {stock_data}')
        feature_x = stock_data['features_x']
        pred = predict_on_df(feature_x)
        result = {
            'prediction': pred,
            'time': stock_data['time'],
            'y': stock_data['y']
        }

        return jsonify(result)

    app.run(debug=True, use_reloader=False, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
