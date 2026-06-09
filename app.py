import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import anthropic
from collections import Counter

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Review Intelligence",
    page_icon="🛒",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #e74c3c;
        margin-bottom: 12px;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #2c3e50; }
    .metric-label { font-size: 13px; color: #7f8c8d; margin-top: 2px; }
    .positive { border-left-color: #2ecc71 !important; }
    .negative { border-left-color: #e74c3c !important; }
    .neutral  { border-left-color: #3498db !important; }
    .kw-chip {
        display: inline-block;
        background: #fdecea;
        color: #c0392b;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 13px;
        margin: 3px;
        font-weight: 500;
    }
    .ai-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border: 1px solid #667eea40;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    merchant_df = pd.read_csv('merchant_sentiment_scores.csv')
    reviews_df  = pd.read_csv('reviews_with_sentiment.csv')
    with open('merchant_keywords.json') as f:
        merchant_kw = json.load(f)
    return merchant_df, reviews_df, merchant_kw

try:
    merchant_df, reviews_df, merchant_kw = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🛒 Olist Review Intelligence")
st.markdown("**NLP + GenAI powered merchant insights** — HuggingFace Transformers · KeyBERT · Claude AI")
st.markdown("---")

if not data_loaded:
    st.error("⚠️ Data files not found. Run `day1_nlp_pipeline.ipynb` first to generate the required CSVs.")
    st.code("Required files:\n  merchant_sentiment_scores.csv\n  reviews_with_sentiment.csv\n  merchant_keywords.json")
    st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Explore merchants")

# Filter options
min_reviews = st.sidebar.slider("Minimum reviews", 5, 50, 10)
sentiment_filter = st.sidebar.selectbox(
    "Show merchants",
    ["All", "Worst performing (bottom 20%)", "Best performing (top 20%)"]
)

filtered = merchant_df[merchant_df['review_count'] >= min_reviews].copy()

if sentiment_filter == "Worst performing (bottom 20%)":
    threshold = filtered['avg_sentiment'].quantile(0.2)
    filtered = filtered[filtered['avg_sentiment'] <= threshold]
elif sentiment_filter == "Best performing (top 20%)":
    threshold = filtered['avg_sentiment'].quantile(0.8)
    filtered = filtered[filtered['avg_sentiment'] >= threshold]

filtered = filtered.sort_values('avg_sentiment')

# Merchant selector
merchant_options = filtered['seller_id'].tolist()
if not merchant_options:
    st.warning("No merchants match the current filters.")
    st.stop()

selected_merchant = st.sidebar.selectbox(
    "Select merchant",
    merchant_options,
    format_func=lambda x: x[:16] + "..."
)

st.sidebar.markdown("---")
api_key = st.sidebar.text_input("Anthropic API key", type="password",
                                 help="Get a free key at console.anthropic.com")


# ── Overview metrics ───────────────────────────────────────────────────────────
st.subheader("📊 Platform overview")

col1, col2, col3, col4 = st.columns(4)
total_reviews   = len(reviews_df)
pct_negative    = (reviews_df['sentiment_label'] == 'NEGATIVE').mean() * 100
avg_score       = reviews_df['review_score'].mean()
total_merchants = len(merchant_df)

with col1:
    st.markdown(f"""<div class="metric-card neutral">
        <div class="metric-value">{total_reviews:,}</div>
        <div class="metric-label">Total reviews analysed</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card negative">
        <div class="metric-value">{pct_negative:.1f}%</div>
        <div class="metric-label">Negative sentiment</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card positive">
        <div class="metric-value">{avg_score:.2f} / 5</div>
        <div class="metric-label">Avg review score</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card neutral">
        <div class="metric-value">{total_merchants}</div>
        <div class="metric-label">Merchants scored</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# ── Merchant deep-dive ─────────────────────────────────────────────────────────
st.subheader(f"🔎 Merchant deep-dive: `{selected_merchant[:20]}...`")

merchant_row = merchant_df[merchant_df['seller_id'] == selected_merchant].iloc[0]
merchant_reviews = reviews_df[reviews_df['seller_id'] == selected_merchant]

col_a, col_b = st.columns([1, 2])

with col_a:
    # Merchant metrics
    sentiment_val = merchant_row['avg_sentiment']
    card_class = "positive" if sentiment_val > 0 else "negative"
    st.markdown(f"""
    <div class="metric-card {card_class}">
        <div class="metric-value">{sentiment_val:.3f}</div>
        <div class="metric-label">Avg sentiment score</div>
    </div>
    <div class="metric-card negative">
        <div class="metric-value">{merchant_row['pct_negative']:.1f}%</div>
        <div class="metric-label">Negative reviews</div>
    </div>
    <div class="metric-card neutral">
        <div class="metric-value">{int(merchant_row['review_count'])}</div>
        <div class="metric-label">Total reviews</div>
    </div>
    <div class="metric-card positive">
        <div class="metric-value">{merchant_row['avg_review_score']:.2f} / 5</div>
        <div class="metric-label">Avg star rating</div>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    # Sentiment distribution pie
    if len(merchant_reviews) > 0:
        label_counts = merchant_reviews['sentiment_label'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ['#2ecc71' if l == 'POSITIVE' else '#e74c3c'
                  for l in label_counts.index]
        wedges, texts, autotexts = ax.pie(
            label_counts.values,
            labels=label_counts.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )
        ax.set_title('Sentiment breakdown for this merchant', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No reviews found for this merchant in the current dataset sample.")


# ── Complaint keywords ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🏷️ Top complaint keywords (KeyBERT)")

keywords = merchant_kw.get(selected_merchant, [])
if keywords and keywords != ['no negative reviews found']:
    chips = "".join([f'<span class="kw-chip">🔴 {kw}</span>' for kw in keywords])
    st.markdown(chips, unsafe_allow_html=True)
else:
    st.info("No keyword data for this merchant — they may have few/no negative reviews (good sign!).")


# ── Sample reviews ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💬 Sample customer reviews")

tab1, tab2 = st.tabs(["😤 Negative reviews", "😊 Positive reviews"])

with tab1:
    neg = merchant_reviews[merchant_reviews['sentiment_label'] == 'NEGATIVE']
    if len(neg) > 0:
        for _, row in neg.head(5).iterrows():
            with st.expander(f"⭐ {int(row['review_score'])} stars — score: {row['sentiment_score']:.2f}"):
                st.write(row['review_comment_message'])
    else:
        st.success("No negative reviews found for this merchant!")

with tab2:
    pos = merchant_reviews[merchant_reviews['sentiment_label'] == 'POSITIVE']
    if len(pos) > 0:
        for _, row in pos.head(5).iterrows():
            with st.expander(f"⭐ {int(row['review_score'])} stars — score: {row['sentiment_score']:.2f}"):
                st.write(row['review_comment_message'])
    else:
        st.info("No positive reviews found in sample.")


# ── GenAI insight generator ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 GenAI merchant insight generator")
st.markdown("Uses Claude AI to generate actionable improvement recommendations based on this merchant's reviews.")

if not api_key:
    st.warning("👈 Enter your Anthropic API key in the sidebar to enable AI insights.")
else:
    if st.button("✨ Generate AI insights for this merchant", type="primary"):
        neg_reviews = merchant_reviews[
            merchant_reviews['sentiment_label'] == 'NEGATIVE'
        ]['review_comment_message'].tolist()[:10]

        kws_text = ", ".join(keywords) if keywords else "not available"

        prompt = f"""You are a business analyst advising an e-commerce merchant on the Olist platform.

Merchant performance data:
- Average sentiment score: {merchant_row['avg_sentiment']:.3f} (range: -1 worst to +1 best)
- Percentage of negative reviews: {merchant_row['pct_negative']:.1f}%
- Average star rating: {merchant_row['avg_review_score']:.2f} / 5
- Total reviews analysed: {int(merchant_row['review_count'])}
- Top complaint keywords (KeyBERT extracted): {kws_text}

Sample negative customer reviews:
{chr(10).join([f'- "{r}"' for r in neg_reviews[:5]])}

Based on this data, provide:
1. **Root cause analysis** — What are the 2-3 core issues driving negative sentiment? Be specific.
2. **Immediate actions** (this week) — 3 concrete steps the merchant can take right now.
3. **Strategic recommendations** (1-3 months) — 2 longer-term improvements.
4. **Risk assessment** — If no action is taken, what is the likely business impact?

Be specific, practical, and business-focused. Format your response clearly with the headers above."""

        with st.spinner("Claude AI is analysing this merchant's data..."):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                ai_response = response.content[0].text

                st.markdown(f"""
                <div class="ai-box">
                    <strong>🤖 Claude AI analysis</strong><br><br>
                    {ai_response.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"API error: {e}. Check your API key and try again.")


# ── Bottom 10 leaderboard ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📉 Worst-performing merchants leaderboard")

bottom10 = merchant_df.nsmallest(10, 'avg_sentiment')[
    ['seller_id', 'review_count', 'avg_sentiment', 'pct_negative', 'avg_review_score']
].copy()
bottom10['seller_id'] = bottom10['seller_id'].str[:16] + '...'
bottom10.columns = ['Merchant ID', 'Reviews', 'Avg Sentiment', '% Negative', 'Avg Stars']
bottom10 = bottom10.reset_index(drop=True)
bottom10.index += 1

st.dataframe(
    bottom10.style
    .background_gradient(subset=['Avg Sentiment'], cmap='RdYlGn')
    .background_gradient(subset=['% Negative'], cmap='Reds')
    .format({'Avg Sentiment': '{:.3f}', '% Negative': '{:.1f}%', 'Avg Stars': '{:.2f}'}),
    use_container_width=True
)

st.markdown("---")
st.caption("Built by Kaja Harshitha · NLP: HuggingFace Transformers + KeyBERT · GenAI: Anthropic Claude API · Data: Olist Brazilian E-Commerce (Kaggle)")
