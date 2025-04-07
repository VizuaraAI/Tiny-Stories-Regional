# 🧪 Language Complexity Analysis: MorphScore & Rényi Entropy

> [!NOTE]
> This folder provides tools for evaluating tokenizers and performing language analysis.
> It consists of two core modules:
> * 🔠 **MorphScore**: Computes a morphological segmentation score for words using either the **Sarvam** or **Sutra** tokenizer. It processes morphological data stored in CSV format for different languages.
> * 📉 **Rényi Entropy**: Calculates the Rényi entropy of tokenized corpora across languages. This serves as an information-theoretic measure of uncertainty and diversity in token distributions, supporting both Sarvam and Sutra tokenizers.

<br> 

## 📦 Requirements
- 🐍 Python 3.7+
- 🧮 NumPy
- 📊 Pandas
- 🤖 Transformers
- ⏳ tqdm

<br> 

## 📁 Folder Structure

```plaintext
.
├── morphscore
│   ├── morphscore.py       # Calculates morphscore using Sarvam or Sutra tokenizers.
│   └── [morph data files]  # CSV files with morphological data for each language.
└── rényi_entropy
    └── renyi.py            # Computes Rényi entropy for training corpora in multiple languages.
```

<br>

## 🔠 MorphScore Evaluation
> _Evaluates tokenizers based on their ability to preserve morphological boundaries across languages_

### ⚙️ Setup
- Navigate to the `morphscore` directory

### 🛠️ Configuration
- In `morphscore.py`, update the CSV file path to point to your morph data file in the desired language

### ▶️ Execution
- The script tokenizes each word using the selected tokenizer (default tokenizer: Sutra)
- You can switch to Sarvam by modifying the tokenizer setting in the code
- It then computes the MorphScore based on segmentation quality
  
```bash
python morphscore.py
```

<br> 

## 📉 Rényi Entropy Analysis
> _Calculates Rényi entropy for tokenized datasets to assess the information density and token distribution quality_

### ⚙️ Setup
- Navigate to the `rényi entropy` directory

### 🛠️ Configuration
- In `renyi.py`, update the dataset paths to point to your tokenized training and validation data.
- Select the tokenizer by setting the tokenizer variable to 'sutra' or 'sarvam'.

### ▶️ Execution
```bash
python renyi.py
``` 



