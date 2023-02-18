# Prediction of Remaining Useful Life (RUL) for Turbofan Engines
This repository contains  code for predicting the remaining useful life of a turbodan engine based on sensor data using a Convolutional Neural Network (CNN) and Support Vector Regression (SVR) model. It also includes a web app built using Streamlit for the front-end and FastAPI for the back-end. Docker is used to deploy the app on an EC2 instance.

# Dataset
The dataset used in this project is a publicly available dataset of sensor data from a turbodan engine. It includes various sensor readings such as temperature, pressure, and vibration. The goal is to predict the remaining useful life of the engine based on this data

#CNN-SVR Model
The CNN and SVR models were trained using the sensor data from the dataset. The CNN model was used to extract features from the sensor data and the SVR model was used to predict the remaining useful life of the engine based on these features. The models were built using Python and the Keras and scikit-learn libraries.
![Picture description](images/CNN_SVR.jpg)

#
