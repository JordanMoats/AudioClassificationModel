import tkinter as tk
from tkinter import ttk, filedialog
import librosa
import numpy as np
import tensorflow as tf
from pygame import mixer
import os

class InstrumentClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Instrument Classifier")
        self.root.geometry("600x400")
        
        # Initialize audio player
        mixer.init()
        
        # Load the model
        try:
            self.model = tf.keras.models.load_model('Best_Instrument_Classifier.keras')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        # Define the label encoder classes in the same order as training
        self.class_names = [
            'Accordion', 'Acoustic_Guitar', 'Banjo', 'Bass_Guitar', 'Clarinet',
            'Cymbals', 'Dobro', 'Drum_set', 'Electro_Guitar', 'Floor_Tom',
            'Harmonica', 'Harmonium', 'Hi_Hats', 'Horn', 'Keyboard',
            'Mandolin', 'Organ', 'Piano', 'Saxophone', 'Shakers',
            'Tambourine', 'Trombone', 'Trumpet', 'Ukulele', 'Violin',
            'cowbell', 'flute', 'vibraphone'
        ]

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create and configure widgets
        self.setup_widgets()

        # Current audio file path
        self.current_audio = None

    def setup_widgets(self):
        # File selection button
        self.select_button = ttk.Button(
            self.main_frame, 
            text="Select Audio File",
            command=self.select_file
        )
        self.select_button.grid(row=0, column=0, pady=10)

        # Display selected file path
        self.file_label = ttk.Label(self.main_frame, text="No file selected")
        self.file_label.grid(row=1, column=0, pady=5)

        # Audio playback controls
        self.playback_frame = ttk.Frame(self.main_frame)
        self.playback_frame.grid(row=2, column=0, pady=10)

        self.play_button = ttk.Button(
            self.playback_frame,
            text="Play",
            command=self.play_audio,
            state='disabled'
        )
        self.play_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(
            self.playback_frame,
            text="Stop",
            command=self.stop_audio,
            state='disabled'
        )
        self.stop_button.grid(row=0, column=1, padx=5)

        # Classify button
        self.classify_button = ttk.Button(
            self.main_frame,
            text="Classify",
            command=self.classify_audio,
            state='disabled'
        )
        self.classify_button.grid(row=3, column=0, pady=10)

        # Results display
        self.result_label = ttk.Label(
            self.main_frame,
            text="Classification result will appear here",
            wraplength=400
        )
        self.result_label.grid(row=4, column=0, pady=10)

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.ogg"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.current_audio = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.play_button.config(state='normal')
            self.stop_button.config(state='normal')
            self.classify_button.config(state='normal')

    def play_audio(self):
        if self.current_audio:
            mixer.music.load(self.current_audio)
            mixer.music.play()

    def stop_audio(self):
        mixer.music.stop()

    def extract_mel_spectrogram(self, audio_file_path):
        try:
            y, sr = librosa.load(audio_file_path)
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=128,
                n_fft=2048,
                hop_length=512
            )
            S_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            return S_db
        except Exception as e:
            print(f"Error extracting spectrogram: {e}")
            return None

    def classify_audio(self):
        if not self.current_audio or not self.model:
            return

        # Extract features
        spectrogram = self.extract_mel_spectrogram(self.current_audio)
        if spectrogram is None:
            self.result_label.config(text="Error processing audio file")
            return

        # Prepare for model input
        spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
        spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension

        # Make prediction
        try:
            prediction = self.model.predict(spectrogram, verbose=0)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx] * 100

            # Get the instrument name from the class index
            predicted_instrument = self.class_names[predicted_class_idx]

            # Show top 3 predictions
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            result_text = "Top 3 Predictions:\n\n"
            
            for idx in top_3_indices:
                instrument = self.class_names[idx]
                conf = prediction[0][idx] * 100
                result_text += f"{instrument}: {conf:.2f}%\n"

            self.result_label.config(text=result_text)
        except Exception as e:
            self.result_label.config(text=f"Error during classification: {str(e)}")

def main():
    root = tk.Tk()
    app = InstrumentClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()