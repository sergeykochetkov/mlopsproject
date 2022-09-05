'''
unittests for lambda-function handler interface
'''

import unittest
from test_predict import load_test_data
from cloud_function.lambda_function import handler, predict_on_df


class TestLocal(unittest.TestCase):
    '''
    unittests for lambda_function.handler entry point
    '''

    def test_inference(self):
        '''
        tests lambda-function outputs in the case of correct input
        '''
        data = load_test_data()
        event = {"request_id": 1,
                 "features_x": data['x']}
        lambda_output = handler(event, None)

        self.assertEqual(lambda_output['request_id'], event['request_id'])
        self.assertEqual(lambda_output['status_code'], 0)
        self.assertEqual(lambda_output['error_msg'], '')

        model_output = predict_on_df(data['x'])

        self.assertEqual(lambda_output['prediction'], model_output)

    def test_exception(self):
        '''
        tests error code and message in case of incorrect input size
        '''
        event = {"request_id": 2,
                 "features_x": [1, 2, 3]}
        lambda_output = handler(event, None)

        self.assertEqual(lambda_output['request_id'], event['request_id'])
        self.assertNotEqual(lambda_output['status_code'], 0)
        self.assertNotEqual(lambda_output['error_msg'], '')


if __name__ == "__main__":
    unittest.main()
