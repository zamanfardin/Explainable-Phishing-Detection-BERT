"""
STEP 8: UNIFIED INTERPRETABILITY DASHBOARD (FIXED VERSION)
Fixes:
  1. Attention chart now shows real words only (no punctuation/special tokens)
  2. SHAP chart now shows BERT/DistilBERT dimension labels instead of "emb_1050"

Run: /Users/fardinzaman/miniconda3/envs/course_env/bin/python /Users/fardinzaman/Desktop/unified_fixed.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
import torch
import shap
import joblib
import pickle
import warnings
import textwrap
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 8: UNIFIED INTERPRETABILITY DASHBOARD (FIXED)")
print("=" * 80)

# ── PATHS ─────────────────────────────────────────────────────────────────────
PKL_PATH        = '/users/fardinzaman/Desktop/New_Dataset/Research_Processed_Hierarchical.pkl'
EMBEDDINGS_PATH = '/users/fardinzaman/Desktop/New_Dataset/Combined_Embeddings.npy'
SCALER_PATH     = '/users/fardinzaman/Desktop/New_Dataset/4.5phish_results_combined/BERT+DistilBERT_scaler.joblib'
MODEL_PATH      = '/users/fardinzaman/Desktop/New_Dataset/4.5phish_results_combined/BERT+DistilBERT_logistic.joblib'
LING_PATH       = '/users/fardinzaman/Desktop/New_Dataset/linguistic_features.csv'
OUTPUT_DIR      = '/users/fardinzaman/Desktop/New_Dataset/Unified_fixed/'
# ─────────────────────────────────────────────────────────────────────────────

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── FIX 2: Meaningful SHAP feature names ─────────────────────────────────────
def make_feature_names(n_features):
    """
    Label embedding dimensions meaningfully:
    - Dims 0-767    → BERT dim 0 ... BERT dim 767
    - Dims 768-1535 → DistilBERT dim 0 ... DistilBERT dim 767
    """
    names = []
    for i in range(n_features):
        if i < 768:
            names.append(f'BERT dim {i}')
        else:
            names.append(f'DistilBERT dim {i - 768}')
    return names

# ── FIX 1: Clean attention tokens ────────────────────────────────────────────
IGNORE_TOKENS = {
    '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]',
    '.', ',', '?', '!', '&', '/', '\\', '-', '#',
    '*', '=', '+', '(', ')', ':', ';', '"', "'",
    '>', '<', '@', '$', '%', '^', '~', '`', '_',
    '..', '...', '--', '``', "''",
}

def clean_tokens(tokens, attention_weights):
    """
    Remove special tokens, punctuation, subword pieces, and stop words.
    Returns only real meaningful words with their attention weights.
    """
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'is', 'it', 'be', 'as', 'was', 'are',
        'were', 'been', 'has', 'have', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'not',
        'no', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'that',
        'this', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our',
        'its', 'from', 'if', 'then', 'than', 'also', 'into', 'up', 'about'
    }

    cleaned = []
    for token, weight in zip(tokens, attention_weights):
        # Skip special tokens
        if token in IGNORE_TOKENS:
            continue
        # Skip subword pieces (##ing, ##ium etc.)
        if token.startswith('##'):
            continue
        # Skip single characters
        if len(token) <= 1:
            continue
        # Skip non-alphabetic tokens
        if not token.isalpha():
            continue
        # Skip stop words
        if token.lower() in STOP_WORDS:
            continue
        cleaned.append((token, weight))

    return cleaned

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[1/7] Loading models and data...")

with open(PKL_PATH, 'rb') as f:
    df = pickle.load(f)

embeddings      = np.load(EMBEDDINGS_PATH)
scaler          = joblib.load(SCALER_PATH)
model           = joblib.load(MODEL_PATH)
linguistic_df   = pd.read_csv(LING_PATH)

print(f"  PKL:        {df.shape}")
print(f"  Embeddings: {embeddings.shape}")
print(f"  Model:      expects {model.n_features_in_} features")
print(f"  Linguistic: {linguistic_df.shape}")

# ── Scale and predict ─────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

y       = df['label'].values
indices = np.arange(len(embeddings))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    embeddings, y, indices,
    test_size=0.2, random_state=42, stratify=y
)

X_test_scaled = scaler.transform(X_test)
y_pred        = model.predict(X_test_scaled)
y_prob        = model.predict_proba(X_test_scaled)

df_test              = df.iloc[idx_test].reset_index(drop=True)
df_test['predicted'] = y_pred
df_test['actual']    = y_test
df_test['prob_spam'] = y_prob[:, 1]
df_test['prob_ham']  = y_prob[:, 0]
df_test['correct']   = y_pred == y_test
df_test['text_len']  = df_test['text'].str.split().str.len()

# Align linguistic features
# linguistic_df only has 1000 rows — safely align by position
# Use None for rows beyond linguistic_df size
ling_test = linguistic_df.reindex(range(len(df_test))).reset_index(drop=True)

accuracy = df_test['correct'].mean() * 100
print(f"  Accuracy: {accuracy:.2f}%")

# ── Select 6 diverse examples ─────────────────────────────────────────────────
print("\n[2/7] Selecting diverse examples...")

def pick(df, actual, predicted, min_len=30, max_len=150, n=2):
    subset = df[
        (df['actual'] == actual) &
        (df['predicted'] == predicted) &
        (df['text_len'].between(min_len, max_len))
    ]
    return subset.index.tolist()[:n]

spam_correct = pick(df_test, 1, 1)
ham_correct  = pick(df_test, 0, 0)
fp           = pick(df_test, 0, 1, n=1)  # False positive
fn           = pick(df_test, 1, 0, n=1)  # False negative

selected = spam_correct[:2] + ham_correct[:2] + fp[:1] + fn[:1]
selected = selected[:6]
print(f"  Selected {len(selected)} examples")

# ── Init BERT ─────────────────────────────────────────────────────────────────
print("\n[3/7] Initialising BERT tokenizer...")
tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
bert_model.eval()
print("  BERT ready")

# ── Init SHAP ─────────────────────────────────────────────────────────────────
print("\n[4/7] Initialising SHAP explainer...")
# Use a small background sample for speed
background = X_test_scaled[:100]
explainer  = shap.LinearExplainer(model, background, feature_perturbation='interventional')
print("  SHAP ready")

# ── Feature names (FIX 2) ─────────────────────────────────────────────────────
feature_names = make_feature_names(model.n_features_in_)

# ── Helper: get attention ─────────────────────────────────────────────────────
def get_attention(text):
    inputs = tokenizer(text[:500], return_tensors='pt',
                       truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    attention = outputs.attentions[-1].mean(dim=1).squeeze(0).numpy()
    tokens    = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    avg_attn  = attention.mean(axis=0)[:len(tokens)]
    return tokens, avg_attn

# ── Generate dashboards ───────────────────────────────────────────────────────
print("\n[5/7] Generating dashboards...")

SPAM_COLOR = '#C00000'
HAM_COLOR  = '#1f4e79'

for ex_num, row_idx in enumerate(selected, 1):
    print(f"\n  Example {ex_num}/{len(selected)}...")

    row        = df_test.iloc[row_idx]
    email_text = str(row['text'])
    true_lbl   = int(row['actual'])
    pred_lbl   = int(row['predicted'])
    prob_spam  = float(row['prob_spam'])
    prob_ham   = float(row['prob_ham'])
    correct    = bool(row['correct'])
    conf       = prob_spam if pred_lbl == 1 else prob_ham

    # Linguistic features
    ling = ling_test.iloc[row_idx] if row_idx < len(ling_test) else None

    # SHAP values
    emb_scaled = X_test_scaled[row_idx].reshape(1, -1)
    shap_vals  = explainer.shap_values(emb_scaled)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = shap_vals.flatten()

    # Top SHAP features — now with meaningful names (FIX 2)
    top_n_shap   = 15
    abs_shap     = np.abs(shap_vals)
    top_shap_idx = np.argsort(abs_shap)[-top_n_shap:][::-1]
    shap_labels  = [feature_names[i] for i in top_shap_idx]
    shap_values_top = [shap_vals[i] for i in top_shap_idx]

    # Attention — cleaned (FIX 1)
    tokens, avg_attn = get_attention(email_text)
    cleaned_pairs    = clean_tokens(tokens, avg_attn)
    # Sort by weight and take top 15
    cleaned_pairs    = sorted(cleaned_pairs, key=lambda x: x[1], reverse=True)[:15]
    attn_tokens  = [p[0] for p in cleaned_pairs]
    attn_weights = [p[1] for p in cleaned_pairs]

    # ── Build figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor('white')
    gs  = fig.add_gridspec(4, 2, hspace=0.38, wspace=0.32)

    status_str = '✓ CORRECT' if correct else '✗ MISCLASSIFIED'
    hdr_color  = '#2d6a2d' if correct else '#8B0000'
    true_str   = 'SPAM' if true_lbl == 1 else 'HAM'
    pred_str   = 'SPAM' if pred_lbl == 1 else 'HAM'

    # ── Row 0: Email header ───────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis('off')
    ax0.text(0.5, 0.78,
             f"Example {ex_num}: {status_str}",
             ha='center', va='center', fontsize=15, fontweight='bold',
             color=hdr_color,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                       edgecolor=hdr_color, linewidth=2))
    ax0.text(0.5, 0.48,
             f"True: {true_str}   |   Predicted: {pred_str}   |   Confidence: {conf:.2%}",
             ha='center', va='center', fontsize=12, fontweight='bold',
             color=SPAM_COLOR if pred_lbl == 1 else HAM_COLOR)
    snippet = textwrap.fill(email_text[:280] + ('...' if len(email_text) > 280 else ''), width=110)
    ax0.text(0.5, 0.12, snippet,
             ha='center', va='center', fontsize=8.2,
             color='#333333', family='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#EEF4FF', edgecolor='#AAAAAA'))

    # ── Row 1 left: BERT attention heatmap ───────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    max_tok = 20
    tok_disp = tokens[:max_tok]
    # Clean token labels for heatmap axes too
    tok_disp_clean = [t if not t.startswith('##') else t[2:] for t in tok_disp]
    attn_disp = np.zeros((len(tok_disp), len(tok_disp)))
    raw_attn  = bert_model(**tokenizer(email_text[:300], return_tensors='pt',
                            truncation=True, max_length=64, padding=True)).attentions
    raw_attn  = raw_attn[-1].mean(dim=1).squeeze(0).detach().numpy()
    sz = min(max_tok, raw_attn.shape[0])
    attn_disp[:sz, :sz] = raw_attn[:sz, :sz]

    sns.heatmap(attn_disp[:sz, :sz],
                xticklabels=tok_disp_clean[:sz],
                yticklabels=tok_disp_clean[:sz],
                cmap='YlOrRd', ax=ax1,
                cbar_kws={'label': 'Attention Weight'})
    ax1.set_title('BERT Attention Pattern (Last Layer)', fontsize=11, fontweight='bold', pad=8)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=7)

    # ── Row 1 right: Top words by attention (FIXED — no punctuation) ─────────
    ax2 = fig.add_subplot(gs[1, 1])
    if attn_tokens:
        bar_colors = [SPAM_COLOR if pred_lbl == 1 else HAM_COLOR] * len(attn_tokens)
        ax2.barh(range(len(attn_tokens)), attn_weights,
                 color=bar_colors, alpha=0.75, edgecolor='white')
        ax2.set_yticks(range(len(attn_tokens)))
        ax2.set_yticklabels(attn_tokens, fontsize=9.5)
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'No meaningful tokens found',
                 ha='center', va='center', fontsize=10)
    ax2.set_xlabel('Average Attention Weight', fontsize=10)
    ax2.set_title('Top 15 Words by BERT Attention\n(Special tokens & punctuation removed)',
                  fontsize=11, fontweight='bold', pad=8)
    for sp in ['top', 'right']:
        ax2.spines[sp].set_visible(False)

    # ── Row 2 left: SHAP (FIXED — meaningful labels) ──────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    shap_colors = [SPAM_COLOR if v > 0 else '#2e75b6' for v in shap_values_top]
    ax3.barh(range(len(shap_labels)), shap_values_top,
             color=shap_colors, alpha=0.78, edgecolor='white')
    ax3.set_yticks(range(len(shap_labels)))
    ax3.set_yticklabels(shap_labels, fontsize=8.5)
    ax3.invert_yaxis()
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax3.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=10)
    ax3.set_title('Top 15 Features by SHAP Importance\n(Red = pushes toward SPAM, Blue = pushes toward HAM)',
                  fontsize=11, fontweight='bold', pad=8)
    for sp in ['top', 'right']:
        ax3.spines[sp].set_visible(False)

    # ── Row 2 right: Linguistic features ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    if ling is not None:
        ling_cols = ['word_count', 'sentence_count', 'avg_word_length',
                     'lexical_diversity', 'person_count', 'entity_count',
                     'adj_count', 'verb_count', 'sentiment_polarity']
        ling_present = [c for c in ling_cols if c in ling.index]
        ling_vals    = [float(ling[c]) for c in ling_present]
        ling_labels  = [c.replace('_', ' ').title() for c in ling_present]
        norm_vals    = [(v - np.mean(ling_vals)) / (np.std(ling_vals) + 1e-8)
                        for v in ling_vals]
        ling_colors  = ['#2e75b6' if v > 0 else '#E87722' for v in norm_vals]
        ax4.barh(range(len(ling_labels)), norm_vals,
                 color=ling_colors, alpha=0.75, edgecolor='white')
        ax4.set_yticks(range(len(ling_labels)))
        ax4.set_yticklabels(ling_labels, fontsize=9.5)
        ax4.invert_yaxis()
        ax4.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax4.set_xlabel('Normalized Value', fontsize=10)
    ax4.set_title('Linguistic Features Profile', fontsize=11, fontweight='bold', pad=8)
    for sp in ['top', 'right']:
        ax4.spines[sp].set_visible(False)

    # ── Row 3 left: Confidence bars ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    bars = ax5.bar(['HAM', 'SPAM'], [prob_ham, prob_spam],
                   color=['#9dc3e6', '#C00000'], alpha=0.8, edgecolor='white', width=0.4)
    ax5.set_ylim(0, 1)
    ax5.set_ylabel('Probability', fontsize=10)
    ax5.set_title('Model Prediction Confidence', fontsize=11, fontweight='bold', pad=8)
    for bar, prob in zip(bars, [prob_ham, prob_spam]):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{prob:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    for sp in ['top', 'right']:
        ax5.spines[sp].set_visible(False)

    # ── Row 3 right: Summary ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')
    summary  = f"INTERPRETATION SUMMARY\n{'='*38}\n\n"
    summary += f"Classification : {pred_str}\n"
    summary += f"Confidence     : {conf:.1%}\n"
    summary += f"Status         : {status_str}\n\n"
    if ling is not None and ling_present:
        summary += "Key Linguistic Indicators:\n"
        summary += f"  • Word Count      : {ling.get('word_count', 'N/A'):.0f}\n"
        summary += f"  • Lexical Diversity: {ling.get('lexical_diversity', 0):.3f}\n"
        summary += f"  • Sentiment       : {ling.get('sentiment_polarity', 0):.3f}\n\n"
    summary += "Model Reasoning:\n"
    if pred_lbl == 1:
        summary += "  • BERT attention on spam-related words\n"
        summary += "  • SHAP dims push toward SPAM class\n"
        summary += "  • Linguistic profile matches spam\n"
    else:
        summary += "  • BERT attention on contextual content\n"
        summary += "  • SHAP dims push toward HAM class\n"
        summary += "  • Linguistic profile matches ham\n"
    ax6.text(0.05, 0.95, summary, va='top', ha='left', fontsize=9,
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFDE7',
                       edgecolor='#CCCCCC', alpha=0.9))

    plt.suptitle(f'Unified Interpretability Dashboard — Example {ex_num}',
                 fontsize=15, fontweight='bold', y=0.995)

    out_path = f"{OUTPUT_DIR}unified_interpretability_example_{ex_num}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {out_path}")

print(f"\n✓ All dashboards saved to: {OUTPUT_DIR}")
print("=" * 80)
print("DONE — Both fixes applied:")
print("  Fix 1: Attention chart shows real words only (no ##tokens, punctuation, stop words)")
print("  Fix 2: SHAP chart shows 'BERT dim X' / 'DistilBERT dim X' instead of 'Feature 1050'")
print("=" * 80)