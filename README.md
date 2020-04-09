# solid-goggles
Machine Learning Examples

## About
This repository provides an overivew of common applications in machine learning. It uses (relatively) small datasets to show how to implement some common applications of deep learning. The following concepts are covered:
* Image classification
* Sentiment classification
* Text generation
* Forecasting

## Setup
To run an application:

1. Create virtual environment
  ```sh
  python3 -m venv
  ```
  
2. Activate it
   ```sh
   source venv/bin/activate
   ``` 
   to activate venv on Linux or  
   ```sh
   venv\Scripts\activate.bat
   ``` 
   to activate venv on Windows
  
3. Install dependencies
  ```sh
  poetry install
  ```
  
4. Run jupyter notebook
  ```sh
  jupyter notebook YOUR_FILE
  ```

where YOUR_FILE can be:
* image_classification.ipynb
* sentiment_classification.ipynb
* text_generation.ipynb
* forecasting.ipynb
