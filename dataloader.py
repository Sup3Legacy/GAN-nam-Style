import os
import torchaudio
from torchaudio import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import librosa
import utils
import numpy as np
import warnings
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

full_tracks = utils.load('../fma_metadata/tracks.csv')
TRACKS = full_tracks[full_tracks['set', 'subset'] <= 'small']
GENRES = utils.load('../fma_metadata/genres.csv')
FEATURES = utils.load('../fma_metadata/features.csv')
ECHONEST = utils.load('../fma_metadata/echonest.csv')

# some files are misclipped, see issue #41 and #8
SHORTER_IDS = [99134, 108925, 133297,  # empty
               98565, 98567, 98569]  # < 2sec


def prepare_fullset(musicSet, is_logmel=True, verbose=True):
    i = 0
    for _id, _ in musicSet.iterrows():
        if _id in SHORTER_IDS:
            continue
        filename = utils.get_audio_path('../fma_small/', _id)
        filename_root = os.path.splitext(filename)[0]
        file_stft = filename_root + "_stft.npy"
        file_logmel = filename_root + "_log_mel.npy"
        log_mel_exists = os.path.isfile(file_logmel)
        stft_exists = os.path.isfile(file_stft)
        if not(stft_exists and is_logmel and log_mel_exists):
            print(f"Music {_id} does not have precomputed features... Computing them now")
            x, sr = librosa.load(filename, sr=None, mono=True)
            if not stft_exists:
                stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
                np.save(file_stft, (stft, sr))
            if is_logmel and not log_mel_exists:
                mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
                log_mel = librosa.amplitude_to_db(mel)
                np.save(file_logmel, (log_mel, sr))
        i += 1
        if verbose and i % 100 == 0:
            print(f"Loaded {i} samples")

class MusicSet(Dataset):
    def __init__(self,
                 musicSet,
                 device=torch.device("cpu")):
        self.is_logmel = True
        self.device = device
        self.ids = [_id for _id, _ in TRACKS.iterrows()]
        for i in SHORTER_IDS:
            self.ids.remove(i)
        self.path_extension = ("_log_mel" if self.is_logmel else "_stft")
        self.path_extension += ".npy"
        # --- Data preconditionning
        # Resampling
        self.SAMPLE_RATE = 22050
        self.resample_1 = transforms.Resample(44100, self.SAMPLE_RATE).to(device)
        self.resample_2 = transforms.Resample(48000, self.SAMPLE_RATE).to(device)
        self.to_spectro = transforms.Spectrogram(n_fft=2048, hop_length=512,
                                                 normalized=True).to(device)
        self.from_spectro = transforms.GriffinLim(n_fft=2048,
                                                  hop_length=512).to(device)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        file_audio = utils.get_audio_path('../fma_small/', _id)
        # data_path = os.path.splitext(file_audio)[0] + self.path_extension
        music, sr = torchaudio.load(file_audio)
        music = music.to(device)
        music = torch.mean(music, dim=0)  # from stereo to mono
        if sr == 44100:
            music = self.resample_1(music)
        elif sr == 48000:
            music = self.resample_2(music)
        elif sr != self.SAMPLE_RATE:
            music = torchaudio.functional.resample(music, sr, self.SAMPLE_RATE)
            print(f"Warning! file {_id} has non expected sampling rate {sr}")
        spectro = self.to_spectro(music)
        # precomputed = torch.from_numpy(precomputed).to(self.device)
        features = FEATURES.loc[_id]
        return (spectro, features, _id)