"""
This module defines the API endpoints for sentiment analysis.

The router handles requests related to sentiment prediction by accepting text input,
preprocessing it, and returning sentiment scores. The sentiment analysis is performed
using a machine learning model loaded in the FastAPI app state.

Endpoints:
    - POST /: Accepts input text and returns the sentiment score.
    
Dependencies:
    - Sentiment analysis model: The model is loaded into `app.state.sentiment_analysis_model`.
    
Error Handling:
    - Raises HTTP 500 errors if the model is not loaded correctly 
    or if an issue occurs during text preprocessing.

Example Usage:
    - To predict sentiment, make a POST request to `/` with a JSON payload like: 
      `{ "texts": "I love this!" }`. The response will contain the sentiment score.
"""

from typing import List

import tensorflow as tf

from fastapi import APIRouter, Request, status, HTTPException
from pydantic import BaseModel

router = APIRouter(
    prefix="",
    tags=["models"],
    responses={404: {"description": "Not found"}}
)

class TextStr(BaseModel):
    """
    A Pydantic Model representing the input text for sentiment analysis.
    
    Attributes:
        texts (str): The input text for sentiment analysis.
    """
    texts: str

def pre_process_text(input_text: List[str]):
    """
    Receives a plain text and convert into tf.data.Dataset 
    using from_tensor_slices to feed into model.
    
    Args:
        input_text (List[str]): List of sentence to be classified.

    Returns:
        tf.data.Dataset:
    """
    # Create a TensorFlow dataset from the list of texts
    dataset = tf.data.Dataset.from_tensor_slices(input_text)
    dataset = dataset.batch(1)  # Adjust the batch size as needed
    return dataset

def prediction(dataset, sent_model, text_len):
    """
    Classify the given dataset and return prediction score.

    Args:
        dataset: tf.data.Dataset from tensor slices.
        sent_model: Loaded pre-trained sentiment analysis BERT model.
        text_len: length of the total input

    Returns:
        predictions.item(): Prediction score

    """

    # Perform prediction using the dataset
    # Specify steps based on the number of texts
    predictions = sent_model.predict(dataset, steps=text_len)

    # Return the prediction score
    return predictions.item()

@router.post("/", status_code=status.HTTP_200_OK)
async def predict_sentiment(request:Request, data: TextStr):
    """
    Predict the sentiment of the input text using a pre-loaded machine learning model.

    This endpoint accepts a request with text data, preprocesses the text, and uses the 
    sentiment analysis model to predict the sentiment scores. If the model is not 
    properly loaded or an error occurs during preprocessing, an HTTP 500 error is raised.

    Args:
        request (Request): The FastAPI request object, used to access the app state which 
                           contains the sentiment analysis model.
        data (TextStr): A data object containing the input texts for sentiment prediction.

    Returns:
        dict: A dictionary containing the sentiment prediction score(s) for the input text(s).
              Example: `{"score": [0.95, 0.10]}`.

    Raises:
        HTTPException: If the sentiment model is not loaded or an error occurs during text 
                       preprocessing, an HTTP 500 response is returned with the error details.
    """
    ml_models = request.app.state.sentiment_analysis_model

    if ml_models is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")

    try:
        processed_text = pre_process_text([data.texts])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    predictions = prediction(processed_text, ml_models, len(data.texts))
    return {"score": predictions}
