"""
sentiment_analysis.py

This module contains the implementation for building a sentiment analysis model
using Tensorflow and pre-trained language model (BERT). The model is desgined to 
classify text inputs positive or negative sentiment in-terms of percentage from 0-100.


Key Features:
- Data Preprocessing
- Model pipeline
- Save model
"""

import tensorflow_hub as hub
import tensorflow_text as text  # pylint: disable=unused-import
import tensorflow as tf

PRE_PROCESS_URL= "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
ENCODER_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

# Load the BERT preprocessing and encoder layers from TensorFlow Hub
bert_preprocess = hub.KerasLayer(PRE_PROCESS_URL)
bert_encoder = hub.KerasLayer(ENCODER_URL)

def build_model():
    """
    Build the model for sentiment analysis.
    
    Returns:
        output_model: Returns the model layer
    """
    # Input layer for text
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)

    # Preprocess the input text using the BERT preprocessing layer
    preprocessed_text = bert_preprocess(text_input)

    # Pass the preprocessed text to the BERT encoder
    encoder_outputs = bert_encoder(preprocessed_text)

    # Use the pooled_output (represents the [CLS] token's representation) for classification
    pooled_output = encoder_outputs['pooled_output']

    # Add a Dense layer for binary classification (positive/negative sentiment)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(pooled_output)

    # Create the Keras model
    output_model = tf.keras.Model(inputs=[text_input], outputs=[output])

    return output_model

# Build and compile the model
model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Check the model structure
model.summary()

# Saving the model
model.save("../saved-model/sent-model")

# Ensure the model is in evaluation mode
model.trainable = False

# Example texts for prediction
texts = ["""I recently stayed at a hotel that was highly disappointing.
         The room was dirty, and the staff were unhelpful and rude.
         Despite requesting multiple times, the issues were never addressed.
         The amenities were outdated, and the overall experience was far below what I expected.
         I would not recommend this place to anyone and will avoid it in the future."""]

# Create a TensorFlow dataset from the list of texts
dataset = tf.data.Dataset.from_tensor_slices(texts)
dataset = dataset.batch(1)  # Adjust the batch size as needed

# Perform prediction using the dataset
# Specify steps based on the number of texts
predictions = model.predict(dataset, steps=len(texts))

# Print the predictions
print(predictions)
