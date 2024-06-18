#  ✺1:11─ Real-time Music Visualizer ───── 4:44 ✺

###  The code combines real-time audio processing with dynamic visualizations written in Python and machine learning models. It involves extracting audio features from WAV files, augmenting the audio data, creating a dataset for visual parameters, training regression models to predict visual parameters from audio features, and dynamically visualizing the audio data using Pygame.

##### Created by Reyna Deyna for educational and entertainment purposes, all music used for training the models is also owned by Reyna Deyna. While this project is not intended for scientific research yet, it functions perfectly for entertainment purposes. Its primary focus is on providing a fun experience rather than serving scientific goals or exceptional visual aesthetics.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Details](#details)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/reynadeyna/reynadeyna_music_visualizer.git
    cd real-time-music-visualizer
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### তততততততততততততততততততততততততততততততততততততততততততত
### This project can be used straight away for visualization. Press Space to exit full screen. Simply run:
```
python real_time_vizualisation.py
```

### If you want to use your own audio to create a dataset and train models, then follow these steps:
### ততততততততততততততততততততততততত


### Audio Feature Extraction and Augmentation

Create a directory in root, /audio_samples, add audio files there. To extract and augment audio features, run:
```
python extract_features.py
```

This will create an augmented dataset saved as data/audio_visual_dataset_augmented.csv.

### Model Training

In the /models and /models_comparison directories, various models are available. However, for this project, the best model has been identified and is utilized in the current script. To train the model, execute the following command:

```
python model_training.py
```

This script trains multiple models to predict visual parameters and saves the best models in the models/ directory.

### Visualization

To start the visualization, run:

```
python real_time_vizualisation.py
```

This script will initialize the visualization window where audio features are dynamically visualized.


## Details
### Feature Extraction

The extract_audio_features function extracts the following features from audio:

    Chroma
    Tempo
    Spectral Centroid
    Spectral Bandwidth
    Spectral Rolloff
    RMS (Root Mean Square)
    Zero Crossing Rate

### Data Augmentation

The augment_audio function performs:

    Pitch Shifting
    Time Stretching
    Adding Noise

### Visualization

The script uses Pygame to create dynamic visualizations based on real-time audio input.

### Acknowledgements

This project utilizes the Librosa library for extracting audio features and employs Pygame for visualization.





