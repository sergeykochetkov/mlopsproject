'''
service loading storck quotes data, sending request to
prediction service and logging response
'''

import time
import requests
from cloud_function.consts import URL
from test_predict import load_test_data


def sign(input_number) -> int:
    '''
    :param input_number: any number
    :return: integer +1 -1 depending on signum of input
    '''
    if input_number > 0:
        return 1
    return -1


if __name__ == "__main__":

    while True:
        data = load_test_data(ticker='AAPL')

        time.sleep(1)

        URL = 'https://bbaaeof3gfmonespgi81.containers.yandexcloud.net/predict'
        headers = {
            "Authorization": "Bearer t1.9euelZrPk56TkJ6dkpaWmcaMjZqMkO3rnpWazs6diovHlc7J"
                             "mpaYmo6Yio_l9PdqFwRn-e8wHSu83fT"
                             "3KkYBZ_nvMB0rvA.mv8D6iWlbOxcP8AyIUsaLaNl_dYbQ_VrwYmWw"
                             "nzwsfouNEoMWeY1ZwgnOSeaC2Sa370rG25gIw9GaLpoF5gRBQ"}
        prediction = requests.post(URL, json=data, headers=headers, timeout=10).json()

        print(prediction)

        input_output = data.update(prediction)
