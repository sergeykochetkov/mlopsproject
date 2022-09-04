'''
integration test for testing end-to-end
send request to docker+flask web service
get response
check response
'''

import time
import requests

from consts import URL
from test_predict import load_test_data

if __name__ == "__main__":
    data = load_test_data()

    time.sleep(2)

    print(URL)
    prediction = requests.post(URL, json=data, timeout=10).json()

    assert prediction['time'] == data['time']
    assert prediction['y'] == data['y']
    assert 'prediction' in prediction
