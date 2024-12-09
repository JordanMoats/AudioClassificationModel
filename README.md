# Musical Instrument Classifier

A deep learning model that classifies musical instruments from audio recordings, with a graphical user interface for easy interaction.

## Features
- Deep learning model trained on spectrograms of musical instrument recordings
- Supports 28 different musical instruments
- Simple GUI interface for audio file selection and classification
- Real-time audio playback
- Shows top 3 predictions with confidence scores

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd musical-instrument-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

If you want to train the model yourself:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/abdulvahap/music-instrunment-sounds-for-classification)

2. Place the dataset in a folder named `music_dataset` in the project root directory

3. Open and run `final.ipynb` in Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```

4. Run all cells up to the "Model Training" section

5. Run the final model training cells (the last model in the notebook):
```python
# Create dataset splits
dataset_splits = create_dataset_splits(file_paths)

generator_data = reset_generators(
    file_paths,
    encoded_labels,
    dataset_splits,
    batch_size=32
)

sample_spectrogram = np.load(file_paths[0])
input_shape = (*sample_spectrogram.shape, 1)

best_model = Sequential([
    Input(shape=input_shape),
    
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((4, 4)),
    BatchNormalization(),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((4, 4)),
    BatchNormalization(),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.25),
    
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(num_classes, activation='softmax')
])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

best_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = best_model.fit(
    generator_data['train_generator'],
    steps_per_epoch=generator_data['steps_per_epoch'],
    validation_data=generator_data['val_generator'],
    validation_steps=generator_data['validation_steps'],
    epochs=20,
    callbacks=[early_stopping],
    verbose=1
)

best_model.save("Best_Instrument_Classifier.keras")
```

## Using the GUI

1. Make sure you have a trained model file named `Best_Instrument_Classifier.keras` in the project directory

2. Run the GUI application:
```bash
python instrument_classifier.py
```

3. Using the interface:
   - Click "Select Audio File" to choose an audio file (.wav, .mp3, or .ogg)
   - Use the "Play" and "Stop" buttons to preview the audio
   - Click "Classify" to analyze the audio
   - The top 3 predictions will be displayed with confidence scores

## Model Performance

The final model achieves:
- Training accuracy: ~95%
- Validation accuracy: ~97%
- Test accuracy: ~96%


## Requirements

- Python 3.8+
- See requirements.txt for full list of dependencies
