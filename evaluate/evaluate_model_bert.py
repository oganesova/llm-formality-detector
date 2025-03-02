import torch
from transformers import BertTokenizer, BertForSequenceClassification
from metric_calculator import MetricsCalculator

import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.train_model_bert import convert_frame_to_dataset_format

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

model_path = "models/bert_formality_classifier/"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
test_file_path = "dataset/test_data.csv"


def load_test_dataset():
    try:
        logging.info("Loading and tokenizing test dataset.")
        test_data = convert_frame_to_dataset_format(test_file_path).map(tokenize, batched=True)
        return test_data
    except Exception as e:
        logging.error(f"Error while loading : {e}")

def load_trained_model():
    try:
        return BertForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        logging.error(f"Error while loading model : {e}")

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
    logging.info("Starting model evaluation.")
    model.eval()
    true_labels = []
    predictions = []
    probabilities = []

    logging.info(f"Evaluating {len(test_dataset_example)} examples.")

    for batch in test_dataset_example:

        logging.debug(f"Processing batch.")

        input_ids = torch.tensor(batch['input_ids']).unsqueeze(0)
        attention_mask = torch.tensor(batch['attention_mask']).unsqueeze(0)
        labels = torch.tensor(batch['labels']).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).item()
            probs = torch.softmax(logits, dim=1).numpy()[0][1]
        logging.debug(f"Batch - True label: {labels.item()}, Predicted: {preds}, Probability: {probs:.4f}")


        true_labels.append(labels.item())
        predictions.append(preds)
        probabilities.append(probs)

    logging.info("Model evaluation complete")
    return true_labels, predictions, probabilities

def print_metrics(true_labels, predictions, probabilities):
    logging.info("Calculating and printing metrics.")
    metrics = MetricsCalculator.calculate_all_metrics(true_labels, predictions, probabilities)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"AUC ROC: {metrics['auc_roc']:.4f}")



if __name__ == "__main__":
    logging.info("Starting evaluation...")
    true_label, prediction, probability = evaluate_model()
    print_metrics(true_label, prediction, probability)
    logging.info("Evaluation process complete.")