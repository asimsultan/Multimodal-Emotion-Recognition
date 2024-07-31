
import os
import torch
import argparse
import librosa
import pandas as pd
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, AdamW, get_scheduler
from datasets import load_dataset
from utils import get_device, preprocess_text_data, preprocess_audio_data, MultimodalDataset

def main(data_path):
    # Parameters
    text_model_name = 'bert-base-uncased'
    audio_model_name = 'facebook/wav2vec2-base-960h'
    max_length = 128
    batch_size = 16
    epochs = 3
    learning_rate = 5e-5

    # Load Dataset
    dataset = pd.read_csv(data_path)

    # Preprocess Data
    text_processor = Wav2Vec2Processor.from_pretrained(audio_model_name)
    audio_processor = Wav2Vec2Processor.from_pretrained(audio_model_name)
    preprocessed_data = preprocess_text_data(dataset, text_processor, max_length)
    preprocessed_data = preprocess_audio_data(preprocessed_data, audio_processor)

    # DataLoader
    train_dataset = MultimodalDataset(preprocessed_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Models
    device = get_device()
    text_model = Wav2Vec2ForSequenceClassification.from_pretrained(text_model_name, num_labels=5)
    audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(audio_model_name, num_labels=5)
    text_model.to(device)
    audio_model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(list(text_model.parameters()) + list(audio_model.parameters()), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training Function
    def train_epoch(text_model, audio_model, data_loader, optimizer, device, scheduler):
        text_model.train()
        audio_model.train()
        total_loss = 0

        for batch in data_loader:
            text_inputs = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            audio_inputs = batch['audio_input_ids'].to(device)
            labels = batch['labels'].to(device)

            text_outputs = text_model(text_inputs, attention_mask=text_attention_mask, labels=labels)
            audio_outputs = audio_model(audio_inputs, labels=labels)
            loss = text_outputs.loss + audio_outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch(text_model, audio_model, train_loader, optimizer, device, lr_scheduler)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss}')

    # Save Models
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    text_model.save_pretrained(model_dir)
    audio_processor.save_pretrained(model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing emotion data')
    args = parser.parse_args()
    main(args.data_path)
