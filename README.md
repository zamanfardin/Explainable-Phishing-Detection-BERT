Hey there 
My name is Fardin Zaman, and I’m currently pursuing my Master’s in Cyber Security (Technical) at the University of Southern Queensland, Australia. My Personal Web: fardinzaman.dev

This repository contains our research project titled “Explainable and Adversarially Robust Phishing Detection using BERT and Explainable AI (XAI)”.
The goal of this project is to develop a trustworthy phishing email detection system that not only detects phishing attempts accurately but also explains why those decisions are made  while testing how resistant the model is to subtle adversarial manipulations.


Phase 1: Data Preprocessing (Completed)

------Loaded and cleaned raw dataset (processed_data.csv).

------Filled missing subject fields and combined subject + body into a unified text column.

------Removed null entries and exported final dataset (Research_Dataset_Final.csv).

------Dataset ready for modeling and analysis.

**Phase 2:** Exploratory Data Analysis (Week 1–2)

Perform EDA to understand text distributions, word frequencies, and class imbalance.

Visualize phishing vs. legitimate patterns (e.g., using WordCloud, TF-IDF, n-grams).

Document key insights about linguistic characteristics.

**Phase 3:** Model Development (Week 3–4)

Load the final dataset and split into train/validation/test sets.

Fine-tune BERT / DistilBERT for phishing detection.

Compare with classical baselines (e.g., Logistic Regression, SVM, LSTM).

Evaluate metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
**
Phase 4:** Explainable AI Integration (Week 5–6)

Apply LIME, SHAP, or Integrated Gradients to interpret BERT predictions.

Visualize influential words and contextual cues driving decisions.

Summarize which linguistic features signal phishing intent.

**Phase 5:** Adversarial Robustness Testing (Week 7)

Implement adversarial perturbations (synonym swaps, paraphrasing, misspellings).

Measure how small edits affect model predictions (robustness score).

Combine explainability with robustness to study model weaknesses.
**
Phase 6:** Multilingual Extension (Week 8–9)

Integrate language detection + translation pipeline for non-English samples.

Re-run classification to test translation-based approach.

Compare performance between monolingual and multilingual setups.

**Phase 7:** Final Evaluation & Optimization (Week 10)

Summarize model performance and visualization results.

Optimize training parameters for accuracy and robustness.

Prepare final comparative report (English vs multilingual, original vs adversarial).

**Phase 8:** Documentation & Publication (Week 11)

Clean up code and add comments for reproducibility.

Write detailed project documentation in GitHub README.

Prepare research paper or poster summarizing results.

Submit to faculty / journal (Q1–Q2 cybersecurity or AI publication).
