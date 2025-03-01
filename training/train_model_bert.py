import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

file_path_train = "dataset/train_data.csv"
file_path_val = "dataset/val_data.csv"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def convert_frame_to_dataset_format(file_path):

    data_df = pd.read_csv(file_path)
    return Dataset.from_pandas(data_df)

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
    return train_dataset, val_dataset

def load_model():
    train_dataset, val_dataset = get_tokenized_datasets()

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

    trainer.train()
    trainer.save_model("models/bert_formality_classifier/")
    print("Model training complete and saved!")

if __name__ == "__main__":
    print("Starting training...")
    load_model()
