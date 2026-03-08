# 🚀 Sentiment Analysis Engine – Amazon Reviews

A production-ready **sentiment analysis application** that classifies Amazon product reviews as positive or negative using machine learning. Process single reviews interactively or analyze entire CSV datasets with 100k+ reviews at scale.

---

## ✨ Features

- **Single Review Analysis**: Instant sentiment classification with confidence scores
- **Bulk Processing**: Handle large CSV files (100k+ rows) with chunked memory-efficient processing
- **Pre-trained ML Model**: XGBoost classifier optimized for Amazon review data
- **Beautiful UI**: Neubrutalism-inspired interface built with Streamlit
- **Data Visualization**: Real-time sentiment distribution charts and statistics
- **Download Results**: Export batch predictions as CSV with sentiment labels and confidence percentages
- **Auto Column Detection**: Intelligently identifies text columns in uploaded CSV files

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Web Framework** | Streamlit |
| **ML Model** | XGBoost (classification) |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Vectorization** | Count Vectorizer (NLP) |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Vercel (Python runtime) |
| **NLP** | NLTK |

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd sentiment-analysis-amazon-reviews
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application

**Local Development:**
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

## 📦 Project Structure

```
sentiment-analysis-amazon-reviews/
├── app.py                          # Main Streamlit application
├── main.ipynb                      # Jupyter notebook (model training/analysis)
├── amazon.tsv                      # Sample dataset (3,152 Amazon Echo reviews)
├── requirements.txt                # Python package dependencies
├── vercel.json                     # Vercel deployment configuration
├── readme.md                       # This file
└── Models/
    ├── model_xgb.pkl             # Pre-trained XGBoost model
    ├── scaler.pkl                # MinMax scaler for feature normalization
    └── countVectorizer.pkl       # Text vectorizer (bag of words)
```

---

## 📊 Dataset

The project includes **amazon.tsv** with 3,152 Amazon Echo reviews containing:

| Column | Description |
|--------|-------------|
| `rating` | Product rating (1-5 stars) |
| `date` | Review date |
| `variation` | Product variant (e.g., Charcoal Fabric) |
| `verified_reviews` | Review text content |
| `feedback` | Verified purchase indicator |

**Sentiment Labels:**
- Ratings 4-5 → **Positive** (1)
- Ratings 1-3 → **Negative** (0)

---

## 💻 How to Use

### Single Review Prediction
1. Paste or type a review in the **"Single Review Prediction"** card
2. Click the **"Analyze 🚀"** button
3. View instant sentiment classification and confidence percentage

Example input:
```
"Amazing product! Works perfectly and arrived quickly. Highly recommend!"
```

### Bulk CSV Processing
1. Upload a CSV file in the **"Bulk CSV Prediction"** card
2. Select the text column containing reviews
3. Click **"Run Bulk Prediction"**
4. View sentiment distribution charts
5. Download results as `sentiment_results.csv` with:
   - Original data
   - `sentiment` column: "Positive" or "Negative"
   - `confidence_%` column: prediction confidence (0-100)

**Supported CSV formats:**
- Headers required
- Auto-detect text columns (looks for: "review", "comment", "feedback", "text", "message")
- Handles 100k+ rows efficiently with chunked processing (10k rows per chunk)

---

## 🤖 Model Details

| Aspect | Details |
|--------|---------|
| **Algorithm** | XGBoost Classifier |
| **Feature Extraction** | Count Vectorizer (bag of words) |
| **Scaling** | MinMax Scaler |
| **Training Data** | Amazon Echo reviews |
| **Classes** | 2 (Positive/Negative) |
| **Input** | Text reviews (string) |
| **Output** | Class label (0/1) + probability score |

### Model Pipeline:
1. Text preprocessing (lowercase, remove special characters)
2. Count vectorization (word frequency)
3. Feature scaling (MinMax normalization)
4. XGBoost prediction + confidence scoring

---

## 🎨 UI/UX Design

The interface features a **Neubrutalism** design aesthetic:

- **Bold typography** with heavy 800-weight fonts
- **Bright color palette**: Yellow (#FFD166), Emerald (#06D6A0), Red (#EF476F), Blue (#118AB2)
- **Black borders** (4-5px) with offset box shadows
- **High contrast** for accessibility
- **Responsive layouts** using Streamlit columns

---

## 📈 Performance Notes

- **Single predictions**: ~100-200ms
- **Bulk processing**: ~50-100ms per 10k-row chunk
- **Memory usage**: Streaming CSV chunks prevents memory overflow on large files
- **Scalability**: Tested with Amazon EC2 and Vercel Python runtime

---

## 🚢 Deployment

### Deploy on Vercel

1. **Prerequisites:**
   - Vercel account ([vercel.com](https://vercel.com))
   - GitHub repository connected

2. **Deploy Steps:**
   ```bash
   npm i -g vercel
   vercel
   ```

3. **Configuration:** Uses `vercel.json` for Python runtime setup

4. **Live URL:** Your app will be available at `<project>.vercel.app`

### Environment Variables
No environment variables required for basic deployment.

---

## 📝 Requirements

Installed packages:
- `pandas` – Data manipulation
- `numpy` – Numerical computing
- `matplotlib` – Plotting & visualization
- `seaborn` – Statistical visualization
- `scikit-learn` – Machine learning utilities
- `xgboost` – Gradient boosting model
- `nltk` – Natural language processing
- `wordcloud` – Word cloud generation
- `streamlit` – Web app framework

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Models not found** | Ensure `Models/` folder exists with pickle files |
| **Column not detected** | Rename your CSV column to include: "review", "comment", "feedback", "text", or "message" |
| **Memory error on large files** | Built-in chunking handles 100k+ rows; files >1GB may need the task to process in parts |
| **Streamlit caching issues** | Clear cache: `streamlit run app.py --logger.level=debug` |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License – see the LICENSE file for details.

---

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)

---

## 👨‍💻 Author - The Dhruva

Created for sentiment analysis and NLP exploration on Amazon product reviews.

---

**Last Updated:** March 2026
