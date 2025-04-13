# Deepfake Duel: Truth vs. Trickery

This repository contains a deep learning project aimed at detecting and classifying deepfakes. The model distinguishes between real and fake images and identifies the image class (human_faces, animals, vehicles) using the Artifact_240K dataset.

## Project Structure
- **data/**: Contains train, validation, and test images.
- **models/**: Contains the model definition and folder to store trained weights.
- **notebooks/**: Notebooks for exploratory data analysis and experiments.
- **outputs/**: Contains logs and plots from training.
- **utils/**: Custom dataset, metrics, and logger utilities.
- **main.py**: Entry point for training/validation.
- **predict.py**: Script for inference and generating predictions.
- **reports/**: Submission report.

## Environment Setup
1. Create and activate a virtual environment:
    ```
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## Running the Code
- **Training:**  
  Run `main.py` to start training:
  ```bash
  python main.py
#   d e e p f a k e - d e t e c t i o n  
 