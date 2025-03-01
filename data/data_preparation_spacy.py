import pandas as pd
import spacy
from datasets import Dataset, load_from_disk
from sklearn.model_selection import train_test_split
import os

nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    doc = nlp(text)
    cleaned_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return cleaned_text

def load_and_preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    df['cleaned_text'] = df.apply(lambda row: preprocess_text(row['formal']) if row.name % 2 == 0 else preprocess_text(row['informal']), axis=1)
    df['label'] = df.apply(lambda row: 1 if row.name % 2 == 0 else 0, axis=1)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    train_data = Dataset.from_pandas(train_df[['cleaned_text', 'label']])
    val_data = Dataset.from_pandas(val_df[['cleaned_text', 'label']])
    test_data = Dataset.from_pandas(test_df[['cleaned_text', 'label']])

    return train_data, val_data, test_data

def save_datasets(train_dataset, test_dataset, val_dataset):
    train_dataset.save_to_disk('dataset/train_spacy_dataset')
    test_dataset.save_to_disk('dataset/test_spacy_dataset')
    val_dataset.save_to_disk('dataset/val_spacy_dataset')

if __name__ == "__main__":
    dataset_path = "dataset/formal_informal_dataset.csv"
    train, val, test = load_and_preprocess_data(dataset_path)
    save_datasets(train, val, test)



