# Text Formality Classification Project Documentation

## Project Structure
```
├── dataset/               # Dataset directory
│   ├── formal_informal_dataset.csv   
│   ├── formal_informal_dataset_small.csv
│   ├── train_data.csv    # Training split for BERT
│   ├── val_data.csv      # Validation split for BERT
│   ├── test_data.csv     # Test split for BERT
│   ├── train_spacy_dataset/  # Training data processed with spaCy for RoBERTa
│   ├── val_spacy_dataset/    # Validation data processed with spaCy for RoBERTa
│   └── test_spacy_dataset/   # Test data processed with spaCy for RoBERTa
├── data/                 # Data processing scripts
│   ├── data_preparation.py     # Basic text cleaning and splitting
│   └── data_preparation_spacy.py # NLP processing with spaCy
├── docs/                  # Documentation
│   ├── project_documentation.md  # Technical documentation
│   └── report.md         # Project journey and methodology
├── evaluate/              # Model evaluation scripts
│   ├── evaluate_model_bert.py    # BERT model evaluation
│   ├── evaluate_model_roberta.py # RoBERTa model evaluation
│   └── metric_calculator.py      # Shared metrics calculation
├── models/               # Saved models
│   ├── bert_formality_classifier/    # BERT model artifacts
│   └── roberta_formality_classifier/ # RoBERTa model artifacts
├── training/            # Training scripts
│   ├── train_model_bert.py    # BERT training script
│   └── train_model_roberta.py # RoBERTa training script
└── requirements.txt     # Project dependencies
```

## Models Used

### BERT Model
- Base model: bert-base-uncased
- Pre-trained on large text corpus
- Efficient at handling context
- Good performance on text classification tasks

### RoBERTa Model
- Base model: roberta-base
- Improved training methodology over BERT
- Better performance on many NLP tasks
- More robust training process

## Dataset Versions

### Full Dataset (formal_informal_dataset.csv)
- Contains 1,400 lines of text
- Suitable for production use
- Provides comprehensive training data

### Test Dataset (formal_informal_dataset_small.csv)
- Contains 37 lines of text
- Designed for quick testing and development
- Useful for validating code changes

## Data Processing Scripts

### data_preparation.py
- Basic text cleaning and dataset splitting
- Implements:
  - CSV file reading
  - Null value handling
  - Duplicate removal
  - Text column renaming
  - Label encoding (0 for informal, 1 for formal)
  - Dataset splitting into train/val/test

### data_preparation_spacy.py
- Advanced NLP processing using spaCy
- Features:
  - Uses 'en_core_web_sm' spaCy model
  - Text lemmatization
  - Stop word removal
  - Punctuation cleaning
  - Hugging Face Dataset format conversion
  - Disk-based dataset storage

## Libraries Used

### Core Libraries
- **pandas**: Data processing and manipulation
- **transformers**: BERT model handling and tokenization
- **torch**: Deep learning framework
- **sklearn**: Evaluation metrics and dataset splitting
- **datasets**: Hugging Face dataset handling
- **sentencepiece**: Tokenizer for text processing
- **accelerate**: Hugging Face Accelerate for faster training
- **spaCy**: Advanced NLP processing and text cleaning
- **en_core_web_sm**: English language model for spaCy

## Data Processing Pipeline

1. **Data Preparation**:
   - Two parallel preprocessing approaches:
     a. Basic preprocessing (data_preparation.py):
        - Dataset cleaning and basic preprocessing
        - Simple text cleaning
        - CSV-based storage
     b. Advanced NLP preprocessing (data_preparation_spacy.py):
        - Lemmatization and advanced text cleaning
        - Stop word and punctuation removal
        - Hugging Face Dataset format
   - Split into train/validation/test sets
   - Label encoding (0 - informal, 1 - formal)

2. **Model Training** (for both BERT and RoBERTa):
   - Model initialization
   - Tokenization
   - Training configuration
   - Model fine-tuning
   - Model saving

3. **Model Evaluation**:
   - Separate evaluation scripts for each model
   - Shared metric calculation - class with static methods
   - Metrics computed:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - ROC AUC
     - Confusion Matrix

## Evaluation

1. **Model-specific Evaluation** (`evaluate_model_bert.py`, `evaluate_model_roberta.py`):
   - Model loading
   - Prediction generation
   - Model-specific processing

2. **Shared Metrics** (`metric_calculator.py`):
   - Centralized metrics calculation
   - Consistent evaluation across models
   - Standardized reporting

## Results Comparison

Both models are evaluated using the same metrics for fair comparison:
- Accuracy scores
- Precision and Recall values
- F1-scores
- ROC AUC curves
- Confusion matrices

## Methods List

### Data Processing Methods
#### data_preparation.py
- `dataset_cleaning(file_path)`: Cleans raw dataset, handles nulls and duplicates, assigns labels
- `split_dataset(data_frame)`: Splits data into train/val/test sets with 80%-10%-10%

#### data_preparation_spacy.py
- `preprocess_text(text)`: Applies spaCy NLP processing, lemmatization, and cleaning
- `load_and_preprocess_data(csv_file)`: Processes data with spaCy and converts to hugging face Dataset
- `save_datasets(train, test, val)`: Saves processed datasets to disk in hugging face format

### Training Methods
#### train_model_bert.py
- `convert_frame_to_dataset_format(file_path)`: Converts CSV to HF Dataset format
- `tokenize(example)`: Applies BERT tokenization with max length 
- `get_tokenized_datasets()`: Prepares tokenized train and validation datasets
- `load_model()`: Initializes and trains BERT model with specified parameters

#### train_model_roberta.py
- `load_and_tokenize_data()`: Loads spaCy-processed data and applies RoBERTa tokenization
- `initialize_model()`: Sets up RoBERTa model for sequence classification
- `setup_training_args()`: Configures training parameters
- `train()`: Executes RoBERTa model training process

### Evaluation Methods
#### evaluate_model_bert.py
- `convert_frame_to_dataset_format(file_path)`: Converts test data to Dataset format
- `load_test_dataset()`: Loads and tokenizes test data
- `load_trained_model()`: Loads trained BERT model
- `tokenize(example)`: Applies BERT tokenization
- `evaluate_model()`: Performs model evaluation and returns predictions
- `print_metrics(true_labels, predictions, probabilities)`: Displays evaluation metrics

#### evaluate_model_roberta.py
- `load_models()`: Loads RoBERTa model and LLM judge
- `load_and_tokenize_dataset(model_tokenizer)`: Prepares test data
- `evaluate_with_llm_judge()`: Evaluates using both metrics and LLM judgment
- `print_metrics(true_labels, predictions, probabilities)`: Displays evaluation results

#### metric_calculator.py
Static methods in `MetricsCalculator` class:
- `get_accuracy_metric(true_labels, predictions)`: Calculates accuracy
- `get_precision_metric(true_labels, predictions)`: Calculates precision
- `get_recall_metric(true_labels, predictions)`: Calculates recall
- `get_f1_score_metric(true_labels, predictions)`: Calculates F1 score
- `get_confusion_metric(true_labels, predictions)`: Generates confusion matrix
- `get_roc_auc_score_metric(true_labels, probabilities)`: Calculates ROC AUC
- `calculate_all_metrics(...)`: Computes all metrics in one call

### Key Parameters
1. Training Parameters:
   - Max sequence length: 128
   - Batch sizes: 16 (BERT), 8 (RoBERTa)
   - Training epochs: 3
   - Save steps: 500

2. Model Configurations:
   - BERT: 'bert-base-uncased'
   - RoBERTa: 'roberta-base'
   - LLM Judge: 'EleutherAI/pythia-410m'

3. Data Processing:
   - Train-Test Split: 80-20
   - Validation-Test Split: 50-50
   - Random seed: 42



