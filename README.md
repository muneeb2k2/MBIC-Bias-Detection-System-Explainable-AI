# 🔍 MBIC Bias Detection System  
## Explainable AI for Text Bias Classification

MBIC Bias Detection System is an Explainable AI (XAI) project designed to detect bias in textual content while providing transparent, human-interpretable explanations for each prediction.

The system combines modern NLP embeddings, advanced machine learning models, class imbalance handling techniques, similarity search indexing, and LIME-based interpretability to deliver both accurate and explainable results.

---

# 🚀 Project Overview

The objective of this project is not only to classify biased vs non-biased text, but to make predictions interpretable and trustworthy by highlighting word-level contributions that influence model decisions.

The system integrates:

- High-dimensional semantic embeddings
- Imbalance-aware training strategies
- Comparative model evaluation
- Explainability using LIME
- Similarity search via FAISS
- Interactive Streamlit deployment

---

# 📊 Dataset & Preprocessing

- Total Samples: 1,800  
  - 1,100 Biased  
  - 555 Non-biased  
  - 145 No Agreement  

### Challenges:
- Small dataset size
- Significant class imbalance
- Minority class underrepresentation

### Feature Engineering:
- Generated 768-dimensional embeddings using `all-mpnet-base-v2`
- Text cleaning and normalization
- Vectorized semantic representations for modeling

---

# ⚖ Handling Class Imbalance

Due to severe imbalance in class distribution, multiple strategies were explored:

- SMOTE (Synthetic Minority Oversampling Technique)
- Class weighting adjustments
- Comparative evaluation across models

These approaches were implemented to improve minority class detection and increase macro-level performance metrics.

---

# 🤖 Modeling & Evaluation

The following classifiers were evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

### Final Model Selection:
**XGBoost**

- Test Accuracy: 74.59%
- F1-Score: 0.6655

Performance limitations were primarily due to dataset size and imbalance rather than model capability.

---

# 🧠 Explainable AI Integration

Explainability was implemented using LIME (Local Interpretable Model-Agnostic Explanations).

### Key Features:
- Word-level contribution highlighting
- Visualization of feature importance per prediction
- Transparent explanation of biased classification decisions

This allows users to understand *why* a piece of text was labeled as biased — increasing trust and accountability.

---

# 🔎 FAISS Similarity Search

To enhance usability and contextual analysis:

- Built a FAISS index for embedding similarity search
- Enabled fast retrieval of semantically similar texts
- Useful for comparative bias inspection

---

# 🌐 Deployment

Deployed using Streamlit with the following capabilities:

- Single text analysis
- Batch text analysis (CSV upload)
- Interactive visualizations
- LIME explanation display
- Downloadable CSV results
- Plotly-based performance graphs

---

# 🛠 Tech Stack

- Python
- scikit-learn
- XGBoost
- Sentence Transformers
- FAISS
- LIME
- Streamlit
- Plotly

---

# 🏗 Project Architecture

```
User Input
     ↓
Text Preprocessing
     ↓
Sentence Embedding (768-dim)
     ↓
SMOTE / Imbalance Handling
     ↓
XGBoost Classifier
     ↓
LIME Explanation Engine
     ↓
FAISS Similarity Search (Optional)
     ↓
Streamlit Web Interface
```

---

# 🎯 Key Technical Contributions

- Implementation of model-agnostic interpretability (LIME)
- Handling of imbalanced NLP classification tasks
- Embedding-based bias detection pipeline
- FAISS similarity indexing integration
- Interactive XAI-based deployment

---

# ⚠ Limitations

- Small dataset size (1,800 samples)
- Severe class imbalance
- Limited minority class examples

Future work could include:
- Larger annotated datasets
- Transformer fine-tuning
- Advanced imbalance-aware deep learning models

---

# 📌 Impact

This project demonstrates how explainable machine learning can be applied to socially sensitive NLP tasks such as bias detection, emphasizing transparency, fairness, and accountability in AI systems.

---

## 👨‍💻 Author

Muhammad Muneeb Zahid  
AI/ML Engineer  
Lahore, Pakistan
