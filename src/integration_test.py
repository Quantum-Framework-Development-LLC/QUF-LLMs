import unittest
from train import train_model
from evaluate import evaluate_model
from deploy import deploy_model

class TestIntegration(unittest.TestCase):

    def test_end_to_end(self):
        # Your test code for end-to-end integration
        self.assertEqual(train_model(), "Training successful")
        self.assertEqual(evaluate_model(), "Evaluation successful")
        self.assertEqual(deploy_model(), "Deployment successful")

if __name__ == '__main__':
    unittest.main()
