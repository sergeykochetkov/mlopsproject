'''
service loading storck quotes data, sending request to
prediction service and logging response
'''

import os
import time
from datetime import datetime
import requests
import pandas as pd
from whylogs.app import Session
from whylogs.app.writers import WhyLabsWriter

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

    writer = WhyLabsWriter()
    session = Session(project="stock-prediction", pipeline="my-pipeline", writers=[writer])

    model_id = os.environ['WHYLABS_MODEL_ID']

    while True:
        data = load_test_data(ticker='AAPL')

        time.sleep(1)

        with session.logger(tags={"datasetId": model_id}, dataset_timestamp=datetime.now()) as ylog:
            prediction = requests.post(URL, json=data, timeout=10).json()

            print(prediction)

            input_output = data.update(prediction)

            ylog.log_dataframe(pd.DataFrame(input_output, index=[0]))
            ylog.log_metrics(targets=[sign(prediction['y'])],
                             predictions=[sign(prediction['prediction'])],
                             scores=[prediction['prediction']],
                             model_type='Classification')
