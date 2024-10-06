"""
main.py

This module serves as the entrypoint for the FastAPI application.

It defines the FastAPI app instance, includes routing, and handles requests
for various endpoints related to the application's functionality.

Key Features:
- Define API routes and their associated request handlers.
- Configures middleware and other app settings.
- Initialize the application and start the server.

Usage:
Run the application using the command:
    fastapi dev main.py
"""
from contextlib import asynccontextmanager

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text  # pylint: disable=unused-import

from fastapi import FastAPI

from .routers import predict

@asynccontextmanager
async def lifespan(apps: FastAPI):
    """
    Loading the ML models such that it will be executed before the application
    starts taking requests, during the startup.
    """
    # Load the ML model
    ml_models = tf.keras.models.load_model("nlp-models/saved-model/sent-model1.h5",
                                            custom_objects={'KerasLayer': hub.KerasLayer},
                                            )

    # Ensure the model is in evaluation mode
    ml_models.trainable = False
    # Store the model in app.state so it can be access globally
    apps.state.sentiment_analysis_model = ml_models
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(predict.router)
