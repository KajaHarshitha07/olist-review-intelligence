# 🛒 Olist Review Intelligence — NLP + GenAI Merchant Insights

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![GenAI](https://img.shields.io/badge/GenAI-Claude%20API-purple)
![License](https://img.shields.io/badge/License-MIT-green)

> **NLP + Generative AI extension of the [Olist E-Commerce Analytics](https://github.com/KajaHarshitha07) project.**  
> Transforms 96K+ unstructured customer reviews into merchant-level sentiment scores, complaint keywords, and AI-generated improvement recommendations — deployed as a live interactive Streamlit app.

---

## 🔗 Live Demo

**👉 [View Live App](https://your-app-url.streamlit.app)** ← *(replace with your Streamlit URL after deploying)*

---

## 📌 Problem statement

The original Olist analytics project identified **R$2.3M in refund exposure** and tracked merchant delivery KPIs — but the 96K+ customer review texts remained unanalysed. This project answers:

- Which merchants have the highest negative sentiment — and **why**?
- What specific complaints are customers raising per merchant?
- What **actionable steps** should underperforming merchants take?

---

## 🧠 Techniques used

| Layer | Technique | Tool / Library |
|-------|-----------|---------------|
| Sentiment classification | DistilBERT (fine-tuned SST-2) | HuggingFace Transformers |
| Keyword extraction | BERT-based keyphrase extraction | KeyBERT |
| Merchant scoring | Aggregation & ranking pipeline | Pandas |
| GenAI recommendations | Prompt engineering + LLM API | Anthropic Claude API |
| Deployment | Interactive multi-page web app | Streamlit Community Cloud |

---

## 📊 Key results

- Processed **5,000+ text reviews** through a transformer-based sentiment pipeline
- Classified reviews as POSITIVE / NEGATIVE with **confidence scores per review**
- Scored **100+ merchants** by average sentiment, % negative reviews, and avg star rating
- Identified merchants with **>60% negative sentiment** — flagged for business intervention
- Extracted top complaint themes: *delivery delays, damaged products, wrong items*
- Generated **per-merchant GenAI recommendations** via Claude API using domain-specific prompts
- Deployed as a **live Streamlit app** — stakeholders can explore any merchant's full NLP profile in real time

---

## 🖥️ App features

```
📊 Platform Overview       — total reviews, % negative, avg rating, merchant count
🔎 Merchant Deep-Dive      — sentiment score, % negative, review count, pie chart
🏷️ Complaint Keywords      — KeyBERT-extracted keyphrases from negative reviews
💬 Sample Reviews          — tabbed view of positive vs negative reviews per merchant
🤖 GenAI Insight Generator — Claude AI generates root cause analysis + action plan
📉 Worst Merchants Board   — ranked leaderboard with colour-coded sentiment heatmap
```

---

## 🗂️ Project structure

```
olist-review-intelligence/
│
├── day1_nlp_pipeline.ipynb        # NLP pipeline — sentiment + keyword extraction
├── app.py                         # Streamlit app — full deployment code
├── requirements.txt               # All dependencies
│
├── merchant_sentiment_scores.csv  # Output: merchant-level sentiment scores
├── merchant_keywords.json         # Output: per-merchant complaint keywords
├── reviews_with_sentiment.csv     # Output: reviews labelled with sentiment
│
└── README.md
```

---

## ⚙️ How to run locally

**1. Clone the repo**
```bash
git clone https://github.com/KajaHarshitha07/olist-review-intelligence
cd olist-review-intelligence
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate data files** *(skip if already present)*
- Open `day1_nlp_pipeline.ipynb` in Jupyter or Google Colab
- Upload Olist CSVs from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- Run all cells — outputs 3 data files automatically

**4. Run the app**
```bash
streamlit run app.py
```

**5. Open in browser**
```
http://localhost:8501
```

---

## 🔑 GenAI setup (optional)

To enable the AI insight generator inside the app:
1. Get a free API key at [console.anthropic.com](https://console.anthropic.com)
2. Enter it in the app sidebar when prompted

---

## 📦 Data source

[Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
96K+ orders · 8 relational tables · 2016–2018 · real anonymised transaction data


---

*Built as part of an end-to-end data science portfolio — from raw transaction data to deployed GenAI-powered insights.*
