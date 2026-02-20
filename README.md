# 📊 Training Feedback Intelligence System

## 🚀 Project Overview

This project was developed during my internship to automate the analysis of training feedback data using AI and NLP. The system processes Excel or CSV feedback forms and generates insights such as question-wise ratings, section-wise performance, sentiment analysis, and summarized suggestions using MiniLM embeddings.

It helps HR teams and training managers quickly understand training effectiveness without manually reading hundreds of responses.

---

## ✨ Features

* Upload Excel or CSV feedback files
* Automatic detection of rating vs text columns
* Question-wise mean and standard deviation
* Section-wise performance grouping using MiniLM similarity
* Sentiment analysis of textual feedback
* Key suggestion extraction
* Interactive dashboard with charts
* Raw data preview

---

## 🧠 Technologies Used

* Python
* Streamlit (Dashboard UI)
* Sentence Transformers – MiniLM model
* Pandas & NumPy (Data processing)
* Scikit-learn (Cosine similarity)
* Plotly (Charts)

---

## 📂 Project Structure

```
training-feedback-ai/
│
├── app.py                 # Streamlit dashboard
├── analyzer_engine.py     # Rating analysis & grouping logic
├── nlp_processor.py       # MiniLM model, sentiment & summarization
├── requirements.txt       # Dependencies
└── README.md
```

---

## ⚙️ How It Works

1. User uploads feedback Excel/CSV file.
2. System detects rating and text columns.
3. MiniLM embeddings group questions into sections.
4. Statistical analysis calculates mean & standard deviation.
5. Sentiment analysis identifies positive/negative feedback.
6. Dashboard displays results instantly.

---

## ▶️ Run Locally

### 1. Clone Repository

```
git clone https://github.com/yourusername/training-feedback-ai.git
cd training-feedback-ai
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run App

```
streamlit run app.py
```

App will open in your browser.

---

## 🌐 Live Demo

(https://training-feedback-ai-7zu97hk8d7ur4syuq9hb6c.streamlit.app/)

---

## 📈 Example Use Cases

* Corporate training evaluation
* Employee satisfaction surveys
* Workshop feedback analysis
* Academic course feedback
* Event feedback automation

---

## 🧩 Future Improvements

* Advanced sentiment model
* Firebase authentication integration
* Export PDF reports
* Real-time analytics dashboard
* Improved accuracy tuning

---

## 👨‍💻 Author

**Bishal Burnwal**
B.Tech Student | AI & Data Science Enthusiast
Developed as part of internship project.

---

## 📜 License

MIT License

---

## ⭐ If you like this project

Please give it a star on GitHub!
