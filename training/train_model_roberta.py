from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, AutoTokenizer, \
    AutoModelForSequenceClassification

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


train_data_path = "dataset/train_spacy_dataset"
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def load_and_tokenize_data():
    try:
        train_dataset = load_from_disk(train_data_path)
        logging.info(f"Loading dataset from: {train_data_path}")

        def tokenize_function(examples):
            return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True, max_length=128)

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        logging.info("Dataset tokenized.")
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        logging.info("Dataset format set to torch.")

        return train_dataset
    except Exception as e:
        logging.error(f"Error in loading/converting data {train_data_path}: {e}")

def initialize_model():
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def train():
    train_dataset = load_and_tokenize_data()

    logging.info("Starting model training.")
    models = initialize_model()
    training_args = TrainingArguments(
        output_dir="models/roberta_formality_classifier/",
        evaluation_strategy="no",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=500,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=models,
        args=training_args,
        train_dataset=train_dataset
    )
    try:
        trainer.train()
        logging.info("Starting model training.")

        models.save_pretrained("models/roberta_formality_classifier/")
        tokenizer.save_pretrained("models/roberta_formality_classifier/")

        logging.info("Model training complete and saved!")
    except Exception as e:
        logging.error(f"Error while train : {e}")

if __name__ == "__main__":
    print("Starting training...")
    train()
