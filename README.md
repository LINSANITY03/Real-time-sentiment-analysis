[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint) ![Tests](https://github.com/LINSANITY03/Real-time-sentiment-analysis/actions/workflows/python-app.yml/badge.svg)

# Real-time-sentiment-analysis

Analyse the user text input in real time and provide sentiment analysis along with visualization, charts.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)

## Project Structure
Here is an overview of the project structure:

```
    app
    ├── internal
    ├── nlp-models
        ├── notebook
        ├── saved-model # ignored
            ├── sent-model
        ├── src
    ├── routers
    ├── tests
    ------- # other files
    ├── README.md
    ├── requirements.txt
```

## Installation

Create a virutalenv and install the required libraries from requirements.txt

```
    virutalenv venv
    venv\Scripts\activate
    pip install -r requirements.txt
```