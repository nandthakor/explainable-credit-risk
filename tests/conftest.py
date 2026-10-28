import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from xai.recourse_engine import load_model_and_data

@pytest.fixture(scope="session")
def model_and_data():
    return load_model_and_data()

@pytest.fixture(scope="session")
def clf_model(model_and_data):
    return model_and_data[0]

@pytest.fixture(scope="session")
def X_train(model_and_data):
    return model_and_data[1]

@pytest.fixture(scope="session")
def y_train(model_and_data):
    return model_and_data[2]

@pytest.fixture(scope="session")
def feature_cols(model_and_data):
    return model_and_data[3]
