# 🌍📚 **StoryEval: Multilingual Story Generation Analysis Toolkit**

> [!NOTE]
> 🛠️ A comprehensive toolkit for evaluating language model-generated stories and analyzing model weights for Small Language Models.

> [!IMPORTANT]
> 📊 Please refer to the Appendix for the detailed results of the statistical analysis discussed below.

<br> 

## 📌 **Overview**

**StoryEval** provides tools to:
1. 🧾 Evaluate generated stories using multiple metrics (BLEU, BERT, METEOR, ROUGE)
2. 🧠 Analyze language model weights and architecture characteristics
3. 🌐 Support multilingual story evaluation (Hindi, Marathi, Bengali, English)

<br> 

## 📁 **Files**

- `📄 analysis/BERT_BLEU_eval.py`: Computes BLEU and BERTScore for generated stories  
- `📄 analysis/METEOR_eval.py`: Calculates METEOR scores for story fluency  
- `📄 analysis/rougek_analysis_v2.py`: Evaluates content overlap using ROUGE metrics  
- `📄 analysis/weights_test.py`: Analyzes trained SLM model weights using WeightWatcher  

<br>

## ⚙️ **Requirements**

```bash
- 🔧 PyTorch  
- 🔧 Transformers  
- 🔧 NLTK  
- 🔧 datasets (Hugging Face)  
- 🔧 rouge_score  
- 🔧 weightwatcher  
- 🔧 matplotlib  
- 🔧 pandas  
```

<br>

## 🚀 **Key Features**

### 📊 Multiple Evaluation Metrics
Comprehensive story quality assessment with:
- 🔵 **BLEU** and 🟣 **BERTScore** for *semantic similarity*
- 🟡 **METEOR** for *fluency and adequacy*
- 🔴 **ROUGE** for *content overlap and structure*

### 🧮 Model Weight Analysis
Inspect trained model weights using **WeightWatcher** to:
- 🚨 Identify potential issues in model architecture  
- 📈 Visualize singular values across layers  
- 📊 Analyze power law distributions in weight matrices  
- 🔍 Track weak rank loss for optimization insights  

### 🌍 Multilingual Support
Works with multiple languages, including:
- 🗣️ Hindi  
- 🗣️ Marathi  
- 🗣️ Bengali  
- 🗣️ English  

<br>

## 🧪 **Usage**

### 📐 Evaluate stories with BLEU and BERT scores
```bash
python analysis/BERT_BLEU_eval.py
```

### ✨ Calculate METEOR scores
```bash
python METEOR_eval.py
```

### 📚 Analyze with ROUGE metrics
```bash
python rougek_analysis_v2.py
```

### 🧠  Analyze trained model weights
```bash
python weights_test.py
```

<br>

## 📊 Visualization
> All scripts automatically generate and save visualizations to help interpret results:

- 📊 Bar charts for BLEU, BERT, and METEOR scores
- 📈 Layer-wise analysis of model weights
- 🔢 Power law exponents across layers
- ⚠️ Warning distribution in model architecture

<br>

## 📦 Dataset Compatibility
> Compatible with TinyStories datasets and regional variants:

- 🇬🇧 `HuggingFace/roneneldan/TinyStories` (English)
- 🇮🇳 `HuggingFace/TinyStories-Regional/hindi-generated_4o-mini_2M`
- 🇮🇳 `HuggingFace/TinyStories-Regional/marathi-generated_4o-mini_2M`
- 🇮🇳 `HuggingFace/TinyStories-Regional/beng-generated_4o-mini_2M`
