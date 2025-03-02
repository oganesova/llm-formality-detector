import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

file_path_train = "dataset/train_data.csv"
file_path_val = "dataset/val_data.csv"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def convert_frame_to_dataset_format(file_path):
    logging.info(f"Converting {file_path} to Hugging Face Dataset.")
    try:
        data_df = pd.read_csv(file_path)
        logging.info(f"Converting {file_path} to Hugging Face Dataset.")

        return Dataset.from_pandas(data_df)
    except Exception as e:
        logging.error(f"Error converting {file_path}: {e}")

def tokenize(example):
    model_inputs = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    model_inputs["labels"] = example["label"]

    return model_inputs

def get_tokenized_datasets():
    train_dataset = convert_frame_to_dataset_format(file_path_train).map(tokenize, batched=True)
    val_dataset = convert_frame_to_dataset_format(file_path_val).map(tokenize, batched=True)
    logging.info("Datasets tokenized.")

    return train_dataset, val_dataset

def load_model():
    train_dataset, val_dataset = get_tokenized_datasets()
    logging.info("Loading and training BERT model.")

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="models/bert_formality_classifier/",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        save_steps=500,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    try:
        logging.info("Starting model training.")
        trainer.train()
        trainer.save_model("models/bert_formality_classifier/")
        logging.info("Model saved. Model training complete and saved!")
    except Exception as e:
        logging.error(f"Error while train : {e}")


if __name__ == "__main__":
    print("Starting training...")
    load_model()
