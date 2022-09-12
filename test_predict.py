'''
unittests for testing flask web service
'''
import unittest
import multiprocessing
import time
import requests

from cloud_function.predict_service import main
from cloud_function.consts import URL
from data import load_test_data



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

        data = load_test_data()
        time.sleep(6)

        print(URL)
        session = requests.Session()
        try:
            prediction = session.post(URL, json=data, timeout=10).json()
            session.close()
        finally:
            session.close()

        thr.terminate()
        thr.join()

        self.assertEqual(prediction['time'], data['time'])
        self.assertIn('prediction', prediction)


if __name__ == "__main__":
    unittest.main()
