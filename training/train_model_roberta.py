from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, AutoTokenizer, \
    AutoModelForSequenceClassification

train_data_path = "dataset/train_spacy_dataset"
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def load_and_tokenize_data():
    train_dataset = load_from_disk(train_data_path)

    def tokenize_function(examples):
        return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train_dataset

def initialize_model():
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def setup_training_args():
    return TrainingArguments(
        output_dir="models/roberta_formality_classifier/",
        evaluation_strategy="no",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=500,
        logging_dir="./logs"
    )

def train():
    train_dataset = load_and_tokenize_data()
    models = initialize_model()
    training_args = setup_training_args()


    trainer = Trainer(
        model=models,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()

    models.save_pretrained("models/roberta_formality_classifier/")
    tokenizer.save_pretrained("models/roberta_formality_classifier/")

    print("Model training complete and saved!")

if __name__ == "__main__":
    print("Starting training...")
    train()
