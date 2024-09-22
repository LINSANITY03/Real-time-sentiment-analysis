"""
This module contains tests for the sentiment analysis API endpoints.

The tests validate the functionality and responses of the FastAPI sentiment prediction 
router. They ensure that the sentiment analysis model is working correctly and that 
errors are handled properly.

Tested Endpoints:
    - POST /: Test the sentiment prediction with various input texts.
    
Testing Framework:
    - The tests are implemented using pytest and FastAPI's TestClient.

Test Cases:
    - Model Not Loaded: Simulate a scenario where the model is not loaded and 
      check for appropriate error handling (HTTP 500).
    - Exception Handling: Test if the API properly raises errors during text preprocessing.

Example:
    To run the tests, use:
    ```
    pytest test_predict.py
    ```
"""

import os

import tensorflow._api.v2.compat.v1 as tf
import tensorflow_hub as hub
import tensorflow_text as text  # pylint: disable=unused-import
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from ...routers import predict


tf.disable_v2_behavior() # pylint: disable=no-member

# Create a test FastAPI app
app = FastAPI()
app.include_router(predict.router)

@pytest.fixture
def client():
    """
    Sets up a test client for the FastAPI application with a loaded sentiment analysis model.

    This function initializes the FastAPI application by loading the sentiment analysis model 
    from the specified directory and sets it in the app's state. It uses the TestClient 
    from FastAPI to facilitate testing the API endpoints.
    
    Yields:
        TestClient: A context manager that provides a test client for making requests to 
        the FastAPI application. This client can be used to test the functionality of the 
        sentiment analysis endpoints.

    Note:
        Ensure that the path to the model is correct and that the necessary dependencies 
        (like TensorFlow and TensorFlow Hub) are installed in the environment.
    """
    model_path = os.path.join(os.path.dirname(__file__), '../../nlp-models/saved-model/sent-model')

    # Set up the mock model in the app state
    app.state.sentiment_analysis_model = tf.keras.models.load_model(
        model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )

    # Use TestClient for testing
    with TestClient(app) as c:
        yield c

def test_predict_sentiment_positive(test_client, caplog):
    """
    Tests the sentiment prediction endpoint for positive sentiment input.

    This test simulates a POST request to the sentiment prediction endpoint with a 
    positive input text ("I love this product!") and verifies that the response 
    is successful (HTTP 200) and includes a sentiment score.

    Parameters:
        test_client (TestClient): The test client instance used to make requests 
                                    to the FastAPI application.
        caplog (pytest.LogCaptureFixture): A fixture for capturing log messages during the test.
    
    Assertions:
        - Asserts that the response status code is 200.
        - Asserts that the response JSON contains a "score" key.
    """
    # Prepare test data
    test_data = {"texts": "I love this product!"}

    # Make a POST request to the predict endpoint
    response = test_client.post("/", json=test_data)

    if response.status_code != 200:
        caplog.set_level("ERROR")
        print(f"Error Response JSON: {response.json()}")  # Capture in logs

    # Check the response
    assert response.status_code == 200, f"Response JSON: {response.json()}"
    assert "score" in response.json()

def test_predict_sentiment_model_not_loaded(test_client):
    """
    Tests the sentiment prediction endpoint when the sentiment analysis model is not loaded.

    This test simulates a scenario where the sentiment analysis model is cleared from 
    the application state, leading to an error when trying to make a prediction. 
    It verifies that the endpoint returns an HTTP 500 error with the appropriate error message.

    Parameters:
        test_client (TestClient): The test client instance used to make requests 
                                    to the FastAPI application.

    Assertions:
        - Asserts that the response status code is 500.
        - Asserts that the response JSON contains the expected error message 
          indicating that the model is not loaded properly.
    """
    # Clear the model from the app state
    client.app.state.sentiment_analysis_model = None

    test_data = {"texts": "This is bad!"}
    response = test_client.post("/", json=test_data)

    # Check the error response
    assert response.status_code == 500
    assert response.json() == {"detail": "Model not loaded properly"}
