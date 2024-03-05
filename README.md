# Object Detection Web Application

## Overview

This web application allows users to perform object detection on images and videos using the Faster R-CNN model. It provides a user-friendly interface for uploading images and videos, and it displays the results with bounding boxes around detected objects. You can also train custom CNN models for Your own personal use and then test there accuracy on the appliaction for an unseen data.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Web Application Structure](#web-application-structure)
- [Model Selection](#model-selection)
- [Data Processing](#data-processing)
- [Flask Application](#flask-application)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/GLITCH-08/OBJECT-DETECTION.git
    ```

2. **Set up a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```


## Usage

1. **Run the web application:**

    ```bash
    python app.py
    ```

2. Open your web browser and go to [http://localhost:5000](http://localhost:5000) to access the application.

3. Choose the desired option (`Image` or `Video`) and upload the corresponding file.

4. View the object detection results displayed on the web page.

5. Or choose to train a custom model and then test in on the web application.

## Web Application Structure

The web application is built using Flask and integrates TensorFlow for object detection. Key components include:

- `app.py`: Main Flask application file.
- `templates/`: HTML templates for different pages.
- `static/`: Static files (CSS, images, etc.).
- `Detector.py`: Module for handling object detection tasks.

## Model Selection

The application supports Faster R-CNN model variants for object detection. You can choose a model based on speed, COCO mAP, and outputs. Refer to the provided list of models and their specifications in the README.

## Data Processing

Image and video processing are handled by the `Detector.py` module. The `createBoundingBox` method is responsible for creating bounding boxes around detected objects. The processing includes loading the model, predicting, and drawing bounding boxes. `Detector.py` includes all the functions for prediction on images and video.

## Flask Application

The web application is ran on flask. You can choose the option to (`Image`, `Video`, `train`, `test`)The training and testing code is in `app.py`. It take a drive link and dowloads the dataset from there for prediction. In the testing part you can upload a image and test the model you trained on it.



