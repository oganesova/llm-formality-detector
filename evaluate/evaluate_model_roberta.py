import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_from_disk
import re
from metric_calculator import MetricsCalculator

model_path = "models/roberta_formality_classifier/"
test_dataset_path = "dataset/test_spacy_dataset"
llm_as_a_judge_model = "EleutherAI/pythia-410m"
llm_judge_text = """
Text: "{text}"
Model Prediction: "{prediction}" (Formal or Informal)
True Label: "{true_label}" (Formal or Informal)
Is the model's prediction correct? Answer 'yes' or 'no' and nothing else.
"""

def load_models():
    model_tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    judge_tokenizer = AutoTokenizer.from_pretrained(llm_as_a_judge_model)
    judge_model = AutoModelForCausalLM.from_pretrained(llm_as_a_judge_model)
    return model_tokenizer, model, judge_tokenizer, judge_model

def load_and_tokenize_dataset(model_tokenizer):
    test_dataset = load_from_disk(test_dataset_path)

    def tokenize_function(examples):
        return model_tokenizer(examples["cleaned_text"], padding="max_length", truncation=True, max_length=128)

    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset.with_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return test_dataset

def evaluate_with_llm_judge(model, model_tokenizer, judge_model, judge_tokenizer, test_dataset):
    model.eval()
    correct_llm_judgments = 0
    total_samples = 0
    true_labels = []
    model_predictions = []
    model_probabilities = []

    with torch.no_grad():
        for batch in test_dataset:
            input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0)
            attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0)
            labels = torch.tensor(batch["label"]).unsqueeze(0)
            text = model_tokenizer.decode(input_ids[0], skip_special_tokens=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).item()
            probs = torch.softmax(logits, dim=1).numpy()[0][1]

            prediction_str = "Formal" if preds == 1 else "Informal"
            true_label_str = "Formal" if labels.item() == 1 else "Informal"

            prompt = llm_judge_text.format(text=text, prediction=prediction_str, true_label=true_label_str)
            judge_inputs = judge_tokenizer(prompt, return_tensors="pt")
            judge_outputs = judge_model.generate(**judge_inputs, max_new_tokens=10)
            judge_response = judge_tokenizer.decode(judge_outputs[0], skip_special_tokens=True).lower()

            match = re.search(r'(yes|no)', judge_response)
            if match:
                judgment = match.group(1)
                if (judgment == "yes" and preds == labels.item()) or (judgment == "no" and preds != labels.item()):
                    correct_llm_judgments += 1
            total_samples += 1
            true_labels.append(labels.item())
            model_predictions.append(preds)
            model_probabilities.append(probs)
    llm_accuracy = correct_llm_judgments / total_samples if total_samples > 0 else 0
    return llm_accuracy, true_labels, model_predictions,model_probabilities


def print_metrics(true_labels, predictions, probabilities):
    metrics = MetricsCalculator.calculate_all_metrics(true_labels, predictions, probabilities)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"AUC ROC: {metrics['auc_roc']:.4f}")

if __name__ == "__main__" :
    model_tokenizer_, model_, judge_tokenizer_, judge_model_ = load_models()
    test_dataset_ = load_and_tokenize_dataset(model_tokenizer_)
    llm_accuracy_,true_label, prediction, probability = evaluate_with_llm_judge(model_, model_tokenizer_, judge_model_, judge_tokenizer_, test_dataset_)
    print(f"LLM Judge Accuracy: {llm_accuracy_:.4f}")
    print_metrics(true_label, prediction, probability)