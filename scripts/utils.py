
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

class MultimodalDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['text_input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_text_data(dataset, tokenizer, max_length):
    tokenized_inputs = tokenizer(
        dataset['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    tokenized_inputs["labels"] = torch.tensor(dataset['label'].tolist())
    return tokenized_inputs

def preprocess_audio_data(dataset, processor):
    audio_data = []
    for path in dataset['audio_path']:
        waveform, sample_rate = librosa.load(path, sr=16000)
        inputs = processor(waveform, sampling_rate=16000, return_tensors='pt', padding=True)
        audio_data.append(inputs.input_values[0])
    dataset['audio_input_ids'] = torch.stack(audio_data)
    return dataset
