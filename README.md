Intelligent Threat Detection: Leveraging AI & NLP for Modern Cybersecurity
Welcome

This repository contains the complete research artefacts for my MSc Cyber Security thesis. The work investigates how modern transformer-based natural language processing models can be applied to phishing email detection and, critically, how their decisions can be made interpretable and trustworthy.

Rather than focusing solely on classification accuracy, this project emphasises interpretability, transparency, and real-world cybersecurity applicability.

About the Project

Phishing remains one of the most prevalent and effective cyberattack vectors. While deep learning models such as BERT demonstrate strong performance, they are often criticised for operating as black boxes. This research directly addresses that limitation by combining:

Transformer-based embeddings (BERT and DistilBERT)

Classical, well-understood machine learning classifiers

Multiple layers of explainable AI (XAI) techniques

The result is a phishing detection pipeline that is both effective and understandable to human analysts.

Research Objectives

Detect phishing emails using transformer-based text representations

Compare BERT, DistilBERT, and a combined hierarchical embedding approach

Evaluate Logistic Regression and Random Forest classifiers on deep embeddings

Explain why an email is classified as phishing or legitimate

Connect model behaviour to human-interpretable linguistic patterns

Methodology Overview
Data Preparation

Email data cleaned and standardised

Binary labels: Phishing (spam) and Legitimate (ham)

Representation Learning

BERT embeddings: rich contextual representations (768 dimensions)

DistilBERT embeddings: lightweight and computationally efficient (768 dimensions)

Combined embeddings: concatenation of BERT and DistilBERT (1536 dimensions)

The combined hierarchical embedding is a central contribution of this thesis.

Model Training

Each embedding strategy is evaluated using two classifiers:

Embedding Type	Logistic Regression	Random Forest
BERT	Yes	Yes
DistilBERT	Yes	Yes
BERT + DistilBERT	Yes	Yes

All trained models and scalers are saved to ensure full reproducibility.

Evaluation

Model performance is assessed using:

Confusion matrices

ROC curves

Precision–recall curves

Detailed classification reports (CSV)

These metrics are particularly relevant for imbalanced phishing datasets and real-world email filtering scenarios.

Explainability and Interpretation

A core strength of this project is its focus on explainability and model transparency.

SHAP Analysis

Global feature importance identifying influential embedding dimensions

Local explanations illustrating how individual predictions are formed

Attention Analysis

Token-level attention visualisations from BERT

Identification of suspicious words, phrases, and contextual cues

Linguistic Feature Analysis

Human-interpretable features such as word count, sentiment polarity, and lexical diversity

Clear linguistic differences between phishing and legitimate emails

Unified Interpretability Framework

Integration of SHAP explanations, attention analysis, and linguistic features

End-to-end explanations for individual email predictions

A primary research contribution of this thesis

Repository Structure
├── data/                    # Embeddings and labels
├── models/                  # Trained models and scalers
├── results/                 # Metrics, plots, and evaluation outputs
├── explainability/          # SHAP and token-level analyses
├── attention_analysis/      # BERT attention visualisations
└── unified_interpretability # Combined interpretability case studies
Key Findings

Hierarchical embeddings improve or stabilise classification performance

Classical machine learning models perform competitively on transformer embeddings

Explainability methods reveal meaningful phishing indicators

Model decisions align with known linguistic and behavioural phishing patterns

Practical Relevance

In cybersecurity, trust and accountability are essential. This research demonstrates that advanced NLP-based detection systems can be accurate while remaining transparent and operationally useful. The proposed approach is suitable for integration into real-world email gateways, SOC environments, and future AI-driven security platforms.

Future Work

Integration with live or real-time email systems

Extension to large language models

Deployment-focused phishing detection pipelines

Multilingual phishing detection and analysis

