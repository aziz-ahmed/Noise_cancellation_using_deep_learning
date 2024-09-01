# Noise_cancellation_using_deep_learning
Noise Reduction Using CNN
This project implements a Convolutional Neural Network (CNN) for reducing noise in audio signals. The network is trained on pairs of clean and noisy audio data, learning to produce cleaner audio outputs.

Table of Contents
Introduction
Installation
Usage
Model Architecture
Training
Results
Contributing
License
Introduction
Noise reduction in audio is crucial for various applications, from improving call quality to enhancing music recordings. This project uses a deep learning approach to perform noise reduction by training a CNN model on a dataset of clean and noisy audio samples.

Installation
To run this project, you need Python and the following libraries:

TensorFlow
NumPy
librosa
Matplotlib
You can install these dependencies using pip:

bash
Copy code
pip install tensorflow numpy librosa matplotlib
Usage
Data Preparation:

Place your clean audio files in the CleanData directory.
Place your noisy audio files in the NoisyData directory.
Run the Notebook:

Open and run the provided Jupyter Notebook (noise_reduction_CNN_5.ipynb).
The notebook will load the data, preprocess it, and train the model.
Model Evaluation:

After training, the notebook visualizes the waveforms of the clean, noisy, and denoised audio.
Model Architecture
The model is built using TensorFlow and consists of convolutional and deconvolutional layers. The architecture is as follows:

Input Layer: Takes in 1D audio signals.
Convolutional Layers: Extract features from the audio signals.
Deconvolutional Layers: Reconstruct the audio signal to reduce noise.
Training
The model is trained using batches of audio data with a mean squared error (MSE) loss function. The training process involves:

Loading and batching the data.
Training the model over a specified number of epochs.
Visualizing the results.
Results
The notebook provides visualizations of the original noisy and clean signals, along with the denoised output from the model. The results demonstrate the model's effectiveness in reducing noise.

Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your changes.
