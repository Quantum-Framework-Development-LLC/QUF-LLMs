import unittest
from train import train_model
from evaluate import evaluate_model

class TestModel(unittest.TestCase):

    def test_train_model(self):
        # Your test code for training
        self.assertEqual(train_model(), "Training successful")

    def test_evaluate_model(self):
        # Your test code for evaluation
        self.assertEqual(evaluate_model(), "Evaluation successful")

if __name__ == '__main__':
    unittest.main()
