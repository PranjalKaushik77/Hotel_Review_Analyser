# 🏨 Hotel Review Sentiment Analyzer

A powerful web application that analyzes hotel customer reviews to extract sentiment insights and uncover service-related strengths and weaknesses.

---

## 🚀 Features

- **Sentiment Analysis**: Classifies each review as Positive, Neutral, or Negative.
- **Service Aspect Extraction**: Identifies mentions of key service areas like:
  - Room Service
  - Food Quality
  - Staff Service
  - Location
  - Amenities
  - Cleanliness
  - Comfort
  - Value for Money
- **Interactive Visualizations**: Pie charts, bar graphs, and histograms powered by Plotly.
- **Multiple Input Modes**:
  - Upload your own CSV of reviews
  - Use built-in sample data
  - Manually enter reviews in-app
- **Downloadable Output**: Export the analysis results to CSV.

---

## 📁 Project Structure

```
hotel-review-analyzer/
│
├── app.py                # Streamlit application
├── requirements.txt      # Required Python libraries
├── current.txt           # Installed libraries snapshot
├── hotel_reviews.csv     # Sample dataset (optional)
├── README.md             # You're reading it!
└── hotel_analyzer_env/   # (Optional) Python virtual environment
```

---

## ⚙️ Installation

### ✅ Option 1: Run Locally (Recommended)
1. **Clone the repo**
   ```bash
   git clone https://github.com/PranjalKaushik77/Hotel_Review_Analyser.git
   cd Hotel_Review_Analyser
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv hotel_analyzer_env
   hotel_analyzer_env\Scripts\activate  # For Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Visit the app** at: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Usage

### 🔹 Option 1: Upload CSV File
- Ensure your CSV has at least this format:
  ```
  review_id,customer_name,rating,review_text
  1,John Doe,5,"The hotel was amazing!"
  ```

### 🔹 Option 2: Use Built-in Sample Data
- Click "Use Sample Data" from the sidebar.

### 🔹 Option 3: Manual Entry
- Paste reviews one per line in the text area.

---

## 📊 Outputs

- **Sentiment Scores** (via TextBlob):
  - Positive: score > 0.1
  - Neutral: score between -0.1 and 0.1
  - Negative: score < -0.1
- **Service Aspect Scores**: Frequency of mentions in each sentiment group
- **Visual Charts**:
  - Sentiment distribution pie chart
  - Service aspects by sentiment bar chart
  - Polarity score histogram

---

## 🧠 NLP Behind the Scenes

- **TextBlob** for sentiment scoring
- **NLTK** for tokenization and text preprocessing
- **Custom rules** for service aspect extraction
- **Plotly** for visuals
- **Streamlit** for web interface

---

## 🔮 Future Enhancements

- Transformer-based sentiment classification (e.g. BERT)
- Real-time API integration for review feeds
- Multi-language support
- Competitor comparison dashboards
- LDA-based topic modeling

---

## 🛠️ Troubleshooting

| Issue                      | Solution                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| App not loading           | Ensure all dependencies are installed (`pip install -r requirements.txt`) |
| NLTK data missing         | App auto-downloads it on first run. If error persists, check internet.   |
| CSV file errors           | Ensure valid headers and UTF-8 encoding                                  |
| MSVC Error on Windows     | Install **Microsoft Build Tools** from: https://visualstudio.microsoft.com/visual-cpp-build-tools/ |

---

## 🤝 Contributors

- [@PranjalKaushik77](https://github.com/PranjalKaushik77)


