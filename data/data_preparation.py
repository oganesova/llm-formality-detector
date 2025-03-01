import pandas as pd
from sklearn.model_selection import train_test_split


def dataset_cleaning(file_path):

    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    informal_df = df[['informal']].copy()
    informal_df.rename(columns={'informal': 'text'}, inplace=True)
    informal_df["label"] = 0

    formal_df = df[['formal']].copy()
    formal_df.rename(columns={'formal': 'text'}, inplace=True)
    formal_df["label"] = 1

    cleaned_df = pd.concat([informal_df, formal_df], ignore_index=True)

    return cleaned_df

def split_dataset(data_frame):

    train, temp = train_test_split(data_frame, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    train['text'] = train['text'].str.strip()
    val['text'] = val['text'].str.strip()
    test['text'] = test['text'].str.strip()
    train.to_csv("dataset/train_data.csv", index=False)
    val.to_csv("dataset/val_data.csv", index=False)
    test.to_csv("dataset/test_data.csv", index=False)

if __name__ == "__main__":
    dataset_path = "dataset/formal_informal_dataset.csv"
    clean = dataset_cleaning(dataset_path)
    split_dataset(clean)
