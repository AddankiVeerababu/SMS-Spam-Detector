📱 **SMS Spam Detection Using Machine Learning**

This project presents a machine learning-based system to detect spam messages in SMS data. It demonstrates a complete machine learning pipeline, from preprocessing and feature extraction to training and evaluating several classification models using Python.

---

📝 **Project Summary**

With spam messages growing more sophisticated, automated detection has become crucial. This project processes an SMS dataset with labeled spam/ham messages, applies natural language processing, and compares several machine learning models to determine which best classifies message types.

---

📌 **Key Highlights**

- ✅ Cleaned and analyzed 5,000+ SMS messages  
- 🧹 Removed stopwords, punctuation, and performed tokenization  
- 🔠 Used TF-IDF vectorization for feature extraction  
- 🧠 Trained multiple ML models: Naive Bayes, SVM, KNN, Decision Tree, LDA, QDA  
- 📈 Evaluated using accuracy, F1-score, and confusion matrices  
- 📊 Compared model performance to determine the best spam classifier

---

📚 **Notebook Workflow**

1. **Library Imports** – `pandas`, `numpy`, `nltk`, `matplotlib`, `sklearn`  
2. **Data Loading** – Imported dataset with SMS text and spam labels  
3. **Text Preprocessing** – Tokenization, stopword removal, TF-IDF transformation  
4. **Exploratory Data Analysis** – Class distributions and message length analysis  
5. **Modeling** – Trained classifiers including:  
   - Naive Bayes  
   - Support Vector Machine (SVM)  
   - K-Nearest Neighbors (KNN)  
   - Decision Tree  
   - Linear Discriminant Analysis (LDA)  
   - Quadratic Discriminant Analysis (QDA)  
6. **Evaluation** – Accuracy, precision, recall, F1-score, confusion matrix  
7. **Insights** – Discussed top-performing models and future improvement areas

---

📊 **Visualizations Include**

- Spam vs. Ham class distribution  
- Histogram of message lengths  
- Confusion matrices for each model  
- Model accuracy comparison

---

🛠️ **Tools & Technologies Used**

| Tool / Library | Purpose |
|----------------|---------|
| **Python** | Core programming |
| **Jupyter Notebook** | Development interface |
| **pandas & numpy** | Data processing |
| **nltk** | Natural language preprocessing |
| **scikit-learn** | Machine learning models |
| **matplotlib & seaborn** | Data visualization |

---

🧾 **File Details**

- `SMS Spam Detector.ipynb` – Complete notebook with analysis and modeling  
- `spam.csv` – Source dataset (e.g., from UCI repository)

