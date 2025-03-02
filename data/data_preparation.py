import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def dataset_cleaning(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded dataset : {file_path}")
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        informal_df = df[['informal']].copy()
        informal_df.rename(columns={'informal': 'text'}, inplace=True)
        informal_df["label"] = 0

        formal_df = df[['formal']].copy()
        formal_df.rename(columns={'formal': 'text'}, inplace=True)
        formal_df["label"] = 1

        cleaned_df = pd.concat([informal_df, formal_df], ignore_index=True)
        logging.info(f"Splitting and Cleaning is complete to dataset : {file_path}")

        return cleaned_df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"Error processing dataset: {e}")

def split_dataset(data_frame, path_train, path_val, path_test):
    try:
        train, temp = train_test_split(data_frame, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        train.to_csv(path_train, index=False)
        val.to_csv(path_val, index=False)
        test.to_csv(path_test, index=False)
        logging.info(f"Saved datasets : {path_train}, {path_val}, {path_test}")
    except Exception as e:
        logging.error(f"Error splitting dataset: {e}")

if __name__ == "__main__":
    dataset_path = "dataset/formal_informal_dataset_small.csv"
    file_train = "dataset/train_data.csv"
    file_val = "dataset/val_data.csv"
    file_test = "dataset/test_data.csv"
    clean = dataset_cleaning(dataset_path)
    split_dataset(clean,file_train,file_val,file_test)
