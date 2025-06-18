import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter, defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import io
import base64

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class HotelReviewAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.service_keywords = {
            'room_service': ['room service', 'housekeeping', 'cleaning', 'towels', 'bedding', 'maintenance'],
            'food_quality': ['food', 'restaurant', 'breakfast', 'dinner', 'meal', 'cuisine', 'menu', 'chef'],
            'staff_service': ['staff', 'reception', 'front desk', 'concierge', 'employee', 'service', 'helpful'],
            'location': ['location', 'nearby', 'walking distance', 'transportation', 'attractions', 'view'],
            'amenities': ['pool', 'gym', 'spa', 'wifi', 'parking', 'facilities', 'amenities'],
            'value_for_money': ['price', 'expensive', 'cheap', 'value', 'worth', 'money', 'cost'],
            'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitized', 'spotless', 'filthy'],
            'comfort': ['comfortable', 'bed', 'pillow', 'quiet', 'noise', 'sleep', 'cozy']
        }
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def get_sentiment_score(self, text):
        """Get sentiment score using TextBlob"""
        if pd.isna(text) or text == "":
            return 0, "neutral"
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return polarity, sentiment
    
    def extract_service_aspects(self, text):
        """Extract service-related aspects from reviews"""
        if pd.isna(text):
            return {}
        
        text_lower = text.lower()
        aspects = {}
        
        for category, keywords in self.service_keywords.items():
            mentions = 0
            for keyword in keywords:
                mentions += len(re.findall(r'\b' + keyword + r'\b', text_lower))
            aspects[category] = mentions
            
        return aspects
    
    def analyze_reviews(self, df, review_column):
        """Main analysis function"""
        results = []
        
        for idx, review in df[review_column].items():
            # Get sentiment
            polarity, sentiment = self.get_sentiment_score(review)
            
            # Extract service aspects
            aspects = self.extract_service_aspects(review)
            
            # Preprocess text
            clean_text = self.preprocess_text(review)
            
            result = {
                'review_id': idx,
                'original_review': review,
                'clean_review': clean_text,
                'sentiment': sentiment,
                'polarity_score': polarity,
                **aspects
            }
            results.append(result)
        
        return pd.DataFrame(results)

def create_visualizations(df_analyzed):
    """Create various visualizations for the analysis"""
    
    # Sentiment Distribution
    fig1 = px.pie(df_analyzed, names='sentiment', title='Sentiment Distribution of Reviews',
                  color_discrete_map={'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#FFD700'})
    
    # Service Aspects Analysis
    service_cols = ['room_service', 'food_quality', 'staff_service', 'location', 
                   'amenities', 'value_for_money', 'cleanliness', 'comfort']
    
    aspect_data = []
    for col in service_cols:
        for sentiment in ['positive', 'negative', 'neutral']:
            avg_mentions = df_analyzed[df_analyzed['sentiment'] == sentiment][col].mean()
            aspect_data.append({
                'aspect': col.replace('_', ' ').title(),
                'sentiment': sentiment,
                'avg_mentions': avg_mentions
            })
    
    aspect_df = pd.DataFrame(aspect_data)
    
    fig2 = px.bar(aspect_df, x='aspect', y='avg_mentions', color='sentiment',
                  title='Average Service Aspect Mentions by Sentiment',
                  color_discrete_map={'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#FFD700'})
    fig2.update_xaxes(tickangle=45)
    
    # Polarity Score Distribution
    fig3 = px.histogram(df_analyzed, x='polarity_score', nbins=30, 
                       title='Distribution of Sentiment Polarity Scores',
                       color_discrete_sequence=['#4CAF50'])
    
    return fig1, fig2, fig3

def generate_insights(df_analyzed):
    """Generate business insights from the analysis"""
    insights = []
    
    # Overall sentiment breakdown
    sentiment_counts = df_analyzed['sentiment'].value_counts()
    total_reviews = len(df_analyzed)
    
    insights.append(f"📊 **Overall Sentiment Analysis:**")
    insights.append(f"- Positive: {sentiment_counts.get('positive', 0)} ({sentiment_counts.get('positive', 0)/total_reviews*100:.1f}%)")
    insights.append(f"- Negative: {sentiment_counts.get('negative', 0)} ({sentiment_counts.get('negative', 0)/total_reviews*100:.1f}%)")
    insights.append(f"- Neutral: {sentiment_counts.get('neutral', 0)} ({sentiment_counts.get('neutral', 0)/total_reviews*100:.1f}%)")
    
    # Service strengths and weaknesses
    service_cols = ['room_service', 'food_quality', 'staff_service', 'location', 
                   'amenities', 'value_for_money', 'cleanliness', 'comfort']
    
    # Calculate average mentions for positive vs negative reviews
    pos_reviews = df_analyzed[df_analyzed['sentiment'] == 'positive']
    neg_reviews = df_analyzed[df_analyzed['sentiment'] == 'negative']
    
    strengths = []
    weaknesses = []
    
    for col in service_cols:
        pos_avg = pos_reviews[col].mean() if len(pos_reviews) > 0 else 0
        neg_avg = neg_reviews[col].mean() if len(neg_reviews) > 0 else 0
        
        if pos_avg > neg_avg and pos_avg > 0.5:
            strengths.append((col.replace('_', ' ').title(), pos_avg))
        elif neg_avg > pos_avg and neg_avg > 0.5:
            weaknesses.append((col.replace('_', ' ').title(), neg_avg))
    
    insights.append(f"\n🟢 **Service Strengths:**")
    if strengths:
        for strength, score in sorted(strengths, key=lambda x: x[1], reverse=True):
            insights.append(f"- {strength} (mentioned {score:.1f} times on average in positive reviews)")
    else:
        insights.append("- No clear strengths identified from the data")
    
    insights.append(f"\n🔴 **Areas for Improvement:**")
    if weaknesses:
        for weakness, score in sorted(weaknesses, key=lambda x: x[1], reverse=True):
            insights.append(f"- {weakness} (mentioned {score:.1f} times on average in negative reviews)")
    else:
        insights.append("- No clear weaknesses identified from the data")
    
    # Overall satisfaction score
    avg_polarity = df_analyzed['polarity_score'].mean()
    satisfaction_level = "High" if avg_polarity > 0.2 else "Medium" if avg_polarity > -0.1 else "Low"
    
    insights.append(f"\n📈 **Overall Customer Satisfaction:** {satisfaction_level}")
    insights.append(f"- Average sentiment score: {avg_polarity:.3f} (Range: -1 to 1)")
    
    return "\n".join(insights)

def main():
    st.set_page_config(page_title="Hotel Review Sentiment Analyzer", layout="wide")
    
    st.title("🏨 Hotel Review Sentiment Analyzer")
    st.markdown("**Analyze customer reviews to understand sentiment and extract service insights**")
    
    # Initialize analyzer
    analyzer = HotelReviewAnalyzer()
    
    # Sidebar for options
    st.sidebar.header("Options")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Upload CSV File", "Use Sample Data", "Enter Manual Reviews"]
    )
    
    df_analyzed = None
    
    if analysis_type == "Upload CSV File":
        st.header("📁 Upload Your Review Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Select review column
            review_column = st.selectbox("Select the column containing reviews:", df.columns)
            
            if st.button("Analyze Reviews"):
                with st.spinner("Analyzing reviews..."):
                    df_analyzed = analyzer.analyze_reviews(df, review_column)
                    st.success("Analysis complete!")
    
    elif analysis_type == "Use Sample Data":
        st.header("📋 Sample Hotel Reviews Analysis")
        
        # Sample data
        sample_reviews = [
            "The hotel was absolutely wonderful! Great location, excellent service, and the food was amazing. The staff were very helpful and friendly.",
            "Terrible experience. The room was dirty, the service was poor, and the food quality was disappointing. Would not recommend.",
            "Good hotel overall. Clean rooms and decent location. The breakfast could be better though.",
            "Amazing stay! The amenities were top-notch, especially the pool and spa. Great value for money.",
            "The hotel is okay. Nothing special but nothing terrible either. Average experience.",
            "Loved the location and the view from our room! Staff service was excellent, very attentive.",
            "Room service was slow and the housekeeping missed our room twice. Food at the restaurant was cold.",
            "Perfect hotel for business travel. Great wifi, comfortable beds, and excellent front desk service.",
            "The price was too high for what we got. Room was small and amenities were limited.",
            "Fantastic experience! Clean, comfortable, great location near attractions. Highly recommend!"
        ]
        
        df_sample = pd.DataFrame({'reviews': sample_reviews})
        
        if st.button("Analyze Sample Reviews"):
            with st.spinner("Analyzing sample reviews..."):
                df_analyzed = analyzer.analyze_reviews(df_sample, 'reviews')
                st.success("Analysis complete!")
    
    elif analysis_type == "Enter Manual Reviews":
        st.header("✍️ Enter Reviews Manually")
        
        manual_reviews = st.text_area(
            "Enter reviews (one per line):",
            height=200,
            placeholder="Enter each review on a new line..."
        )
        
        if manual_reviews and st.button("Analyze Manual Reviews"):
            reviews_list = [review.strip() for review in manual_reviews.split('\n') if review.strip()]
            df_manual = pd.DataFrame({'reviews': reviews_list})
            
            with st.spinner("Analyzing reviews..."):
                df_analyzed = analyzer.analyze_reviews(df_manual, 'reviews')
                st.success("Analysis complete!")
    
    # Display results if analysis is complete
    if df_analyzed is not None:
        st.header("📊 Analysis Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Visualizations", "Detailed Results", "Business Insights"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_reviews = len(df_analyzed)
                st.metric("Total Reviews", total_reviews)
            
            with col2:
                avg_sentiment = df_analyzed['polarity_score'].mean()
                st.metric("Average Sentiment Score", f"{avg_sentiment:.3f}")
            
            with col3:
                positive_pct = (df_analyzed['sentiment'] == 'positive').sum() / total_reviews * 100
                st.metric("Positive Reviews", f"{positive_pct:.1f}%")
            
            # Sentiment breakdown
            st.subheader("Sentiment Breakdown")
            sentiment_counts = df_analyzed['sentiment'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"✅ Positive: {sentiment_counts.get('positive', 0)}")
            with col2:
                st.error(f"❌ Negative: {sentiment_counts.get('negative', 0)}")
            with col3:
                st.warning(f"⚪ Neutral: {sentiment_counts.get('neutral', 0)}")
        
        with tab2:
            st.subheader("📈 Data Visualizations")
            
            fig1, fig2, fig3 = create_visualizations(df_analyzed)
            
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            st.subheader("📋 Detailed Analysis Results")
            
            # Display options
            show_original = st.checkbox("Show original reviews")
            show_clean = st.checkbox("Show preprocessed text")
            
            display_df = df_analyzed[['review_id', 'sentiment', 'polarity_score']].copy()
            
            if show_original:
                display_df['original_review'] = df_analyzed['original_review']
            if show_clean:
                display_df['clean_review'] = df_analyzed['clean_review']
            
            # Add service aspects
            service_cols = ['room_service', 'food_quality', 'staff_service', 'location', 
                           'amenities', 'value_for_money', 'cleanliness', 'comfort']
            for col in service_cols:
                display_df[col] = df_analyzed[col]
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download results
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="hotel_review_analysis.csv",
                mime="text/csv"
            )
        
        with tab4:
            st.subheader("💡 Business Insights & Recommendations")
            insights = generate_insights(df_analyzed)
            st.markdown(insights)
            
            st.subheader("🎯 Actionable Recommendations")
            st.markdown("""
            **Based on the analysis, consider these action items:**
            
            1. **Focus on strengths**: Leverage your top-performing service areas in marketing
            2. **Address weaknesses**: Create improvement plans for frequently criticized aspects
            3. **Monitor sentiment trends**: Regular analysis to track improvement over time
            4. **Staff training**: Use insights to focus training on problem areas
            5. **Service standards**: Update SOPs based on customer feedback patterns
            """)

if __name__ == "__main__":
    main()