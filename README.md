# Snoring Detection using CNN 💤

A deep learning-based project to detect snoring sounds using Convolutional Neural Networks (CNNs). This repository contains code, datasets, and model configurations to train and deploy a snoring detection model, which can be useful for sleep analysis applications and monitoring sleep-related disorders.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Snoring detection is a crucial aspect of diagnosing sleep disorders, such as sleep apnea. This project leverages Convolutional Neural Networks (CNNs) to classify audio data as either snoring or non-snoring sounds. By analyzing audio features, the model aims to accurately distinguish between snoring and other background sounds, providing a basis for further sleep quality analysis.

## Features

- Detects snoring from audio recordings using CNNs.
- Trained on audio spectrograms for high accuracy in sound classification.
- Supports real-time snoring detection via microphone input.
- Includes a pretrained model for quick deployment.

## Dataset

The dataset consists of audio recordings of snoring and non-snoring sounds, converted to spectrogram images for input into the CNN model. Datasets like [AudioSet](https://research.google.com/audioset/) or publicly available snoring sound datasets can be used as a starting point. 

Audio files are preprocessed to create spectrograms using tools like Librosa or Python’s Wave library.

## Requirements

- Python 3.6 or later
- TensorFlow or PyTorch
- Librosa
- OpenCV
- Pandas
- Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/patelandpatel/Snoring-Detection-CNN.git
   cd Snoring-Detection-CNN
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the audio dataset and save it to the `data/` directory. Preprocess the audio files to create spectrograms if needed.

## Usage

1. **Data Preprocessing**  
   Convert audio files into spectrograms:
   ```python
   python preprocess_audio.py --input_dir data/audio_files --output_dir data/spectrograms
   ```

2. **Train the Model**  
   Train the CNN model on your local machine:
   ```bash
   python train.py --dataset data/spectrograms --epochs 50 --batch_size 32
   ```

3. **Evaluate the Model**  
   After training, evaluate the model on a test dataset:
   ```bash
   python evaluate.py --model saved_model.h5 --test_data data/test_spectrograms
   ```

4. **Run Inference**  
   Use the model to detect snoring in new audio clips:
   ```bash
   python predict.py --audio_file path_to_audio.wav --model saved_model.h5
   ```

## Model Architecture

The model is a Convolutional Neural Network (CNN) designed for image-like input. Audio files are transformed into spectrograms, which are used as input images for the CNN. Typical layers include:

- Convolutional layers for feature extraction from spectrograms.
- Pooling layers for downsampling.
- Dense layers for classification.

### Training Parameters

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Batch Size: 32
- Learning Rate: 0.001

## Results

The model achieves high accuracy in detecting snoring sounds from the test dataset. Here are some sample results:

| Class         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| Snoring       | 0.92      | 0.95   | 0.935    |
| Non-Snoring   | 0.94      | 0.90   | 0.92     |

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Create a pull request.

Please follow the [code of conduct](CODE_OF_CONDUCT.md) and adhere to the coding guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
