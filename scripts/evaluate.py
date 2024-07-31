
import torch
import argparse
import pandas as pd
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from utils import get_device, preprocess_text_data, preprocess_audio_data, MultimodalDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main(model_path, data_path):
    # Load Models and Processor
    text_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)

    # Device
    device = get_device()
    text_model.to(device)
    audio_model.to(device)

    # Load Dataset
    dataset = pd.read_csv(data_path)
    preprocessed_data = preprocess_text_data(dataset, processor, max_length=128)
    preprocessed_data = preprocess_audio_data(preprocessed_data, processor)

    # DataLoader
    eval_dataset = MultimodalDataset(preprocessed_data)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=False)

    # Evaluation Function
    def evaluate(text_model, audio_model, data_loader, device):
        text_model.eval()
        audio_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                text_inputs = batch['text_input_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)
                audio_inputs = batch['audio_input_ids'].to(device)
                labels = batch['labels'].to(device)

                text_outputs = text_model(text_inputs, attention_mask=text_attention_mask)
                audio_outputs = audio_model(audio_inputs)
                preds = (torch.argmax(text_outputs.logits, dim=-1) + torch.argmax(audio_outputs.logits, dim=-1)) // 2
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return accuracy, precision, recall, f1

    # Evaluate
    accuracy, precision, recall, f1 = evaluate(text_model, audio_model, eval_loader, device)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned models')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing validation data')
    args = parser.parse_args()
    main(args.model_path, args.data_path)
