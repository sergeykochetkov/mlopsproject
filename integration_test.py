'''
integration test for testing end-to-end
send request to docker+flask web service
get response
check response
'''

import time
import requests

from cloud_function.consts import URL
from data import load_test_data

if __name__ == "__main__":
    data = load_test_data()

    time.sleep(1)

    prediction = requests.post(URL, json=data, timeout=10).json()

    assert prediction['time'] == data['time']
    assert 'prediction' in prediction
