## Basics ##
import time
import os
import numpy as np

## Audio Preprocessing ##
import pyaudio
import wave
import librosa
from scipy.stats import zscore

## Time Distributed CNN ##
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import LSTM


class speechEmotionRecognition:
    """
    Speech Emotion Recognition
    """

    def __init__(self, subdir_model=None):
        if subdir_model is not None:
            self._model = self.build_model()
            self._model.load_weights(subdir_model)

        self._emotion = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Neutral', 5: 'Sad', 6: 'Surprise'
        }

        # Target mel length so that frame(mel) -> exactly 5 frames (win_size=128, win_step=64)
        # nb_frames = 1 + int((T - 128)/64) = 5  => T in [384, 447]
        self._target_mel_T = 384

    # -------------------------
    # Helpers (NEW)
    # -------------------------
    def _safe_audio(self, y: np.ndarray) -> np.ndarray:
        """Clean NaN/Inf, normalize, and avoid silent audio issues."""
        y = np.asarray(y, dtype=np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # If almost silent, keep as-is but avoid dividing by ~0
        energy = float(np.mean(np.abs(y)))
        if energy < 1e-4:
            # Return as-is; upstream will likely mark low confidence / uncertain
            return y

        max_val = float(np.max(np.abs(y)))
        if max_val > 0:
            y = y / max_val

        return y

    def _safe_zscore_1d(self, x: np.ndarray) -> np.ndarray:
        """Z-score without NaNs when std is ~0."""
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        mu = float(np.mean(x))
        sigma = float(np.std(x))
        if sigma < 1e-8:
            return x - mu
        return (x - mu) / sigma

    def _fix_mel_time(self, mel: np.ndarray) -> np.ndarray:
        """
        Force mel spectrogram time dimension to self._target_mel_T.
        mel shape: (128, T)
        """
        mel = np.asarray(mel, dtype=np.float32)
        mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)

        T = mel.shape[1]
        target = self._target_mel_T

        if T == target:
            return mel

        if T < target:
            pad_width = target - T
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=mel.min())
            return mel

        # If longer, crop (center crop works well)
        start = (T - target) // 2
        return mel[:, start:start + target]

    # -------------------------
    # Voice recording
    # -------------------------
    def voice_recording(self, filename, duration=5, sample_rate=16000, chunk=1024, channels=1):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk
        )

        frames = []
        print('* Start Recording *')
        stream.start_stream()

        start_time = time.time()
        while (time.time() - start_time) < duration:
            # Avoid overflow crashing on some machines
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
        print('* End Recording *')

        # Write wav properly
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    # -------------------------
    # Mel-spectrogram
    # -------------------------
    def mel_spectrogram(self, y, sr=16000, n_fft=512, win_length=256, hop_length=128,
                        window='hamming', n_mels=128, fmax=4000):

        y = self._safe_audio(y)

        # STFT power spectrogram
        S = np.abs(
            librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)
        ) ** 2

        mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels, fmax=fmax)
        mel = librosa.power_to_db(mel, ref=np.max)

        # Force consistent time length
        mel = self._fix_mel_time(mel)

        return np.asarray(mel, dtype=np.float32)

    # -------------------------
    # Framing
    # -------------------------
    def frame(self, y, win_step=64, win_size=128):
        nb_frames = 1 + int((y.shape[2] - win_size) / win_step)

        frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size), dtype=np.float32)
        for t in range(nb_frames):
            frames[:, t, :, :] = np.copy(y[:, :, (t * win_step):(t * win_step + win_size)]).astype(np.float32)

        return frames

    # -------------------------
    # Model
    # -------------------------
    def build_model(self):
        K.clear_session()

        input_y = Input(shape=(5, 128, 128, 1), name='Input_MELSPECT')

        y = TimeDistributed(Conv2D(64, (3, 3), (1, 1), padding='same'), name='Conv_1_MELSPECT')(input_y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D((2, 2), (2, 2), padding='same'), name='MaxPool_1_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT')(y)

        y = TimeDistributed(Conv2D(64, (3, 3), (1, 1), padding='same'), name='Conv_2_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_2_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_2_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D((4, 4), (4, 4), padding='same'), name='MaxPool_2_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_2_MELSPECT')(y)

        y = TimeDistributed(Conv2D(128, (3, 3), (1, 1), padding='same'), name='Conv_3_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_3_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_3_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D((4, 4), (4, 4), padding='same'), name='MaxPool_3_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_3_MELSPECT')(y)

        y = TimeDistributed(Conv2D(128, (3, 3), (1, 1), padding='same'), name='Conv_4_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_4_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_4_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D((4, 4), (4, 4), padding='same'), name='MaxPool_4_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_4_MELSPECT')(y)

        y = TimeDistributed(Flatten(), name='Flat_MELSPECT')(y)
        y = LSTM(256, return_sequences=False, dropout=0.2, name='LSTM_1')(y)
        y = Dense(7, activation='softmax', name='FC')(y)

        return Model(inputs=input_y, outputs=y)

    # -------------------------
    # Prediction (FIXED)
    # -------------------------
    def predict_emotion_from_file(self, filename, chunk_step=16000, chunk_size=49100,
                                 predict_proba=False, sample_rate=16000,
                                 conf_threshold=0.55, smooth=True):

        # Load audio
        y, sr = librosa.core.load(filename, sr=sample_rate, offset=0.5)
        y = self._safe_audio(y)

        # If very silent: return uncertain
        if float(np.mean(np.abs(y))) < 1e-4:
            # Return a single "Uncertain" block
            return [["Neutral"], np.array([0.0])]

        # Chunking
        chunks = self.frame(y.reshape(1, 1, -1), chunk_step, chunk_size)
        chunks = chunks.reshape(chunks.shape[1], chunks.shape[-1])  # (num_chunks, chunk_size)

        # Robust z-normalization per chunk (prevents NaNs)
        y_norm = np.asarray([self._safe_zscore_1d(c) for c in chunks], dtype=np.float32)

        # Mel spectrogram per chunk (fixed time length)
        mel_spect = np.asarray([self.mel_spectrogram(sig, sr=sr) for sig in y_norm], dtype=np.float32)

        # Time distributed framing -> should become (num_chunks, 5, 128, 128)
        mel_spect_ts = self.frame(mel_spect)  # (N, 5, 128, 128)

        # Build X (N, 5, 128, 128, 1)
        X = mel_spect_ts[..., np.newaxis].astype(np.float32)

        # Predict
        proba = self._model.predict(X, verbose=0)  # (N, 7)

        if predict_proba:
            return [proba, None]

        # Convert to labels with confidence threshold
        idx = np.argmax(proba, axis=1)
        conf = np.max(proba, axis=1)

        labels = []
        for i, c in zip(idx, conf):
            if float(c) < conf_threshold:
                labels.append("Neutral")
            else:
                labels.append(self._emotion.get(int(i), "Neutral"))

        # Optional smoothing across chunks (majority vote)
        if smooth and len(labels) >= 3:
            smoothed = labels.copy()
            for i in range(1, len(labels) - 1):
                window = [labels[i - 1], labels[i], labels[i + 1]]
                smoothed[i] = max(set(window), key=window.count)
            labels = smoothed

        # Timestamp (seconds)
        timestamp = np.concatenate([[chunk_size], np.ones((len(labels) - 1)) * chunk_step]).cumsum()
        timestamp = np.round(timestamp / sample_rate).astype(np.float32)

        return [labels, timestamp]

    # -------------------------
    # CSV export
    # -------------------------
    def prediction_to_csv(self, predictions, filename, mode='w'):
        with open(filename, mode) as f:
            if mode == 'w':
                f.write("EMOTIONS\n")
            for emotion in predictions:
                f.write(str(emotion) + "\n")
