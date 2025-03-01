import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from metric_calculator import MetricsCalculator


model_path = "models/bert_formality_classifier/"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
test_file_path = "dataset/test_data.csv"

def convert_frame_to_dataset_format(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return Dataset.from_pandas(df)

def load_test_dataset():
    test_data = convert_frame_to_dataset_format(test_file_path).map(tokenize, batched=True)
    return test_data

def load_trained_model():
    return BertForSequenceClassification.from_pretrained(model_path)

def tokenize(example):
    model_inputs = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    model_inputs["labels"] = example["label"]
    return model_inputs

def evaluate_model():
    test_dataset_example = load_test_dataset()
    model = load_trained_model()
    model.eval()
    true_labels = []
    predictions = []
    probabilities = []

    for batch in test_dataset_example:
        input_ids = torch.tensor(batch['input_ids']).unsqueeze(0)
        attention_mask = torch.tensor(batch['attention_mask']).unsqueeze(0)
        labels = torch.tensor(batch['labels']).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).item()
            probs = torch.softmax(logits, dim=1).numpy()[0][1]

        true_labels.append(labels.item())
        predictions.append(preds)
        probabilities.append(probs)

    return true_labels, predictions, probabilities

def print_metrics(true_labels, predictions, probabilities):
    metrics = MetricsCalculator.calculate_all_metrics(true_labels, predictions, probabilities)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"AUC ROC: {metrics['auc_roc']:.4f}")



if __name__ == "__main__":
    print("Starting evaluation...")
    true_label, prediction, probability = evaluate_model()
    print_metrics(true_label, prediction, probability)