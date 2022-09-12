'''
service loading stock quotes data, sending request to
prediction service and logging response
'''

import time
import argparse
import requests
from data import load_test_data


def sign(input_number) -> int:
    '''
    :param input_number: any number
    :return: integer +1 -1 depending on signum of input
    '''
    if input_number > 0:
        return 1
    return -1


def parse_args():
    '''
    :return: arguments from command line
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--yc_token', type=str,
                        help='authorization token, for obtaining run "yc iam create-token"')
    parser.add_argument('--yc_container_url', type=str,
                        help='url for calling yandex.cloud serverless container')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    while True:
        data = load_test_data(ticker='AAPL')

        time.sleep(1)

        headers = {
            "Authorization": f"Bearer {args.yc_token}"}
        prediction = requests.post(args.yc_container_url,
                                   json=data, headers=headers, timeout=10).json()

        print(prediction)

        input_output = data.update(prediction)
