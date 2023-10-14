#!/bin/bash
echo "Running Unit Tests"
python ./src/test_model.py

echo "Running Integration Tests"
python ./src/integration_test.py
