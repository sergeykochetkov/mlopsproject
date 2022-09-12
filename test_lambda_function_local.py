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
        event = {"time": 10, "request_id": 1,
                 "features_x": data['features_x']}
        lambda_output = handler(event, None)

        self.assertEqual(lambda_output['request_id'], event['request_id'],
                         msg=f'lambda output={lambda_output}')

        self.assertEqual(lambda_output['status_code'], 0,
                         msg=f'lambda output={lambda_output}')
        self.assertEqual(lambda_output['error_msg'], '',
                         msg=f'lambda output={lambda_output}')

        model_output = predict_on_df(data['features_x'])

        self.assertEqual(lambda_output['prediction'], model_output)

    def test_exception(self):
        '''
        tests error code and message in case of incorrect input size
        '''
        event = {"time": 10, "request_id": 2,
                 "features_x": [1, 2, 3]}
        lambda_output = handler(event, None)

        self.assertEqual(lambda_output['request_id'], event['request_id'],
                         msg=f'lambda output={lambda_output}')
        self.assertNotEqual(lambda_output['status_code'], 0,
                            msg=f'lambda output={lambda_output}')
        self.assertNotEqual(lambda_output['error_msg'], '',
                            msg=f'lambda output={lambda_output}')


if __name__ == "__main__":
    unittest.main()
