# 🧉 Alex the Bartender

Alex the Bartender is an interactive AI bartender that uses facial emotion recognition and speech interaction to engage with users. The system integrates a facial emotion recognition model and communicates with the Furhat robot (or its virtual environment) to deliver personalized, expressive conversations.

## 🚀 Demo

🎥 Watch the Demo: Click here to view on Google Drive

## 🧠 Project Overview

This project consists of two main parts:

Facial Emotion Recognition — A deep learning model trained on the DiffusionFER dataset to detect human emotions.
Speech Emotion Recognition — Using Gemini
Interactive Bartender — The Furhat-powered conversational system that reacts dynamically to recognized emotions.

## Setup: 

1. Create a conda environment with the latest python version.
2. Use the requirements.txt and spec-file.txt to install necessary dependencies.

Facial emotion recognition: Model training
1. Put the DiffusionFER dataset in the submission folder
2. Run the train_face_detection.py file: python train_face_detection.py

Interact with Furhat:
1. Start Furhat virtual environment and remote API.
2. Run bartender.py: python bartender.py
