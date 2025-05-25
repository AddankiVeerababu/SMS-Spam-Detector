📱 **SMS Spam Detection Using Machine Learning – Final Project (2025)**

This project presents a machine learning-based system to detect spam messages in SMS data. It was developed as part of an academic final project and demonstrates the full ML workflow, from data preprocessing and feature extraction to model training and evaluation.

---

📝 **Project Summary**

Spam messages continue to be a major issue in digital communication. Automating spam detection helps filter unwanted texts and improves user safety. In this project, we explore a labeled SMS dataset and apply text preprocessing techniques and machine learning algorithms to classify messages as **"spam"** or **"ham"**.

---

📌 **Key Highlights**

- ✅ Cleaned and preprocessed 5,000+ SMS messages  
- 🧹 Removed punctuation, stopwords, and performed tokenization  
- 🔠 Applied text vectorization using TF-IDF  
- 🧠 Trained models: Naive Bayes, Logistic Regression, SVM  
- 📈 Evaluated using accuracy, confusion matrix, and classification report  
- 🧪 Compared multiple models to identify the best performer

---

📚 **Notebook Workflow**

1. **Library Imports** – Used `pandas`, `sklearn`, `nltk`, `matplotlib`, etc.  
2. **Data Loading** – Imported labeled SMS spam dataset  
3. **Text Preprocessing** – Tokenization, stopword removal, TF-IDF vectorization  
4. **Exploratory Analysis** – Visualized message lengths and class distributions  
5. **Modeling** – Trained multiple classifiers (MultinomialNB, Logistic Regression, SVM)  
6. **Evaluation** – Used accuracy, precision, recall, F1-score, and confusion matrix  
7. **Final Output** – Deployed model pipeline with prediction function

---

📊 **Visualizations Include**

- Class distribution (spam vs. ham)  
- Histogram of message lengths  
- Confusion matrix plots  
- Model performance comparison

---

🛠️ **Tools & Technologies Used**

| Tool / Library | Purpose |
|----------------|---------|
| **Python** | Programming language |
| **Jupyter Notebook** | Development environment |
| **pandas & numpy** | Data handling |
| **nltk** | Natural language processing |
| **scikit-learn** | Machine learning pipeline |
| **matplotlib & seaborn** | Data visualization |

---

🧾 **File Details**

- `SMS Spam Detector.ipynb` – Notebook with the full workflow  
- *(Optional)* `spam.csv` – SMS labeled dataset (public UCI dataset)

---

🚀 **How to Run**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-detector.git
   cd sms-spam-detector
