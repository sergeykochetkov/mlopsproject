'''
unittests for testing flask web service
'''
import unittest
import multiprocessing
import time
import mlflow
import requests


from predict import main, RUN_ID, URL
from data import load_dataset


class TestWebService(unittest.TestCase):
    '''
    Tests for client-server interactions
    '''
    def setUp(self) -> None:
        pass

    def test_predict(self):
        '''
        run client sending request to flask server.
        flask server run model on received data and send json with prediction to client back
        '''
        thr = multiprocessing.Process(target=main)
        thr.start()
        run = mlflow.get_run(RUN_ID)

        data = load_dataset(ticker='AAPL', length=int(run.data.params['length']),
                            period=run.data.params['period'], interval=run.data.params['interval'])
        last_time, feature_x, responce_y = data[0][-1], data[1][-1], data[2][-1]
        data = {'time': str(last_time), "y": float(responce_y), "x": list(feature_x.astype(float))}

        time.sleep(2)

        print(URL)
        prediction = requests.post(URL, json=data, timeout=10).json()

        thr.terminate()
        thr.join()

        self.assertEqual(prediction['time'], data['time'])
        self.assertEqual(prediction['y'], data['y'])
        self.assertIn('prediction', prediction)


if __name__ == "__main__":
    unittest.main()
