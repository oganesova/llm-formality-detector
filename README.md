# Improving Writing Assistance at JetBrains AI
# Text Formality Classifier

![image](https://github.com/user-attachments/assets/86498349-54dc-4ee3-8257-33c34681a67e)

A project for classifying texts as formal or informal using BERT and RoBERTa models.

## Models Implemented

1. **BERT** (bert-base-uncased)
   - Traditional BERT architecture
   - Proven performance on text classification
   - Located in `models/bert_formality_classifier/`

2. **RoBERTa** (roberta-base)
   - Enhanced training methodology
   - Improved performance metrics
   - Located in `models/roberta_formality_classifier/`

## Requirements

- Python 3.8+
- pip (Python package manager)

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/oganesova/llm-formality-detector.git
   cd llm-formality-detector
   ```

2. Set up the environment and install dependencies:
```bash

python -m venv venv

venv\Scripts\activate

source venv/bin/activate

pip install -r requirements.txt
```
## Run Data Preparation scripts
3. First run BERT data preparation, after that RoBERTa. It will save everything in dataset dir 

```bash
python data/data_preparation.py

python data/data_preparation_spacy.py
```

## Training Models
Please run this scripts to train models . I try to push this on GitHub, but it almost 9GB.
You can train either or both models:
All models will save in models/ dir 

4. Train BERT:

```bash
python training/train_model_bert.py
```

5. Train RoBERTa:

```bash
python training/train_model_roberta.py
```

## Model Evaluation

Evaluate each model's performance:

6. BERT evaluation:

```bash
python evaluate/evaluate_model_bert.py
```

7. RoBERTa evaluation:

```bash
python evaluate/evaluate_model_roberta.py
```


## Dataset Options

The project supports two dataset versions:
- `formal_informal_dataset.csv`: Full dataset (1,400 lines)
- `formal_informal_dataset_small.csv`: Same dataset , but small version , my computer cant handle 1400 lines (37 lines)


## Project Structure

```
├── dataset/             # Dataset files
├── data/               # Processed and intermediate data
├── docs/               # Documentation
│   ├── project_documentation.md
│   └── report.md
├── evaluate/           # Evaluation scripts
│   ├── evaluate_model_bert.py
│   ├── evaluate_model_roberta.py
│   └── metric_calculator.py
├── models/             # Saved model artifacts
│   ├── bert_formality_classifier/
│   └── roberta_formality_classifier/
├── training/          # Training scripts
│   ├── train_model_bert.py
│   └── train_model_roberta.py
├── venv/              # Python virtual environment
├── requirements.txt   # Project dependencies
```

## Development Environment

The project is set up with:
- IntelliJ IDEA configuration (`.idea/` directory)
- Python virtual environment (`venv/` directory)
- IntelliJ IDEA module configuration (`llm-formality-detector-task.iml`)

## Documentation

1. `docs/project_documentation.md`: Technical details and implementation
2. `docs/report.md`: Project methodology and research process
3. `requirements.txt`: All dependencies with versions

## Model Results

Models are saved in:
- BERT: `models/bert_formality_classifier/`
- RoBERTa: `models/roberta_formality_classifier/`

Each model directory contains:
- Model weights
- Configuration files
- Tokenizer files

## Performance Metrics

Both models are evaluated using:
- Accuracy
- Precision/Recall
- F1-score
- ROC AUC
- Confusion Matrix

Detailed results can be found in the evaluation outputs. 
