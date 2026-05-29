# Advanced-Multi-Label-Toxic-Comment-Classification-using-NLP-Machine-Learning

COMPANY : CODTECH IT SOLUTIONS PVT.LTD 

NAME : Mohana Srinivasulu 

INTERN ID : CITS1567 

DOMAIN : Machine Learning 

DURATION : 6 Weeks 

MENTOR : Neela Santhosh Kumar

# 🧠 Advanced Multi-Label Toxic Comment Classification using NLP & Machine Learning

## 📌 Project Overview

The **Advanced Multi-Label Toxic Comment Classification System** is a complete end-to-end Natural Language Processing (NLP) and Machine Learning project developed using Python, Scikit-learn, Hugging Face datasets, Streamlit, Matplotlib, and Seaborn.

The objective of this project is to automatically identify and classify toxic online comments into multiple harmful categories simultaneously. The system uses advanced text preprocessing, TF-IDF feature extraction, multiple Machine Learning algorithms, model comparison techniques, and a professional web dashboard for real-time predictions.

This project demonstrates the complete Machine Learning lifecycle from data collection to deployment and is designed as an industry-level internship project.

---

# 🎯 Problem Statement

Online platforms such as social media websites, gaming platforms, forums, and discussion communities receive millions of user comments daily. Many of these comments contain harmful language including toxic content, threats, insults, and hate speech.

Manual moderation becomes difficult because:

* Large volume of comments
* Multiple harmful categories in a single comment
* Real-time moderation requirements
* Human moderation limitations

The goal of this project is to build an intelligent AI-based text classification system capable of automatically detecting multiple toxicity labels from user comments using NLP and Machine Learning techniques.

---

# 📖 Objectives of the Project

The major objectives of this project are:

* To understand Natural Language Processing concepts
* To work with large real-world text datasets
* To preprocess and clean textual data
* To perform Exploratory Data Analysis (EDA)
* To apply TF-IDF Vectorization for feature extraction
* To implement Multi-Label Classification
* To train and compare multiple Machine Learning models
* To select the best-performing model automatically
* To evaluate models using advanced metrics
* To develop an interactive Streamlit dashboard
* To deploy an end-to-end Machine Learning system

---

# 📂 Dataset Information

This project uses the **Jigsaw Toxic Comment Classification Dataset** from Hugging Face and Kaggle.

Dataset Source:

https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge

The dataset contains Wikipedia comments labeled into multiple toxicity categories.

---

# 📊 Dataset Labels

| Label         | Description                    |
| ------------- | ------------------------------ |
| toxic         | General toxic comment          |
| severe_toxic  | Highly toxic comment           |
| obscene       | Obscene language               |
| threat        | Threatening language           |
| insult        | Insulting comment              |
| identity_hate | Hate speech targeting identity |

---

# 📈 Dataset Size

| Property     | Value                         |
| ------------ | ----------------------------- |
| Total Rows   | 159,000+                      |
| Problem Type | Multi-Label Classification    |
| Domain       | NLP / Toxic Comment Detection |
| Dataset Type | Real-world text data          |

---

# 🛠 Technologies Used

| Category              | Technology           |
| --------------------- | -------------------- |
| Programming Language  | Python               |
| IDE                   | Visual Studio Code   |
| NLP                   | TF-IDF Vectorization |
| Machine Learning      | Scikit-learn         |
| Dashboard Development | Streamlit            |
| Data Visualization    | Matplotlib, Seaborn  |
| Dataset Source        | Hugging Face         |
| Model Saving          | Joblib               |

---

# ⚙️ Project Workflow

The project follows a complete end-to-end Machine Learning workflow.

---

# 1️⃣ Data Collection

The dataset was collected from Hugging Face using the `datasets` library. The dataset contains real-world toxic comments and multiple label annotations.

The dataset was automatically downloaded and converted into a Pandas DataFrame for preprocessing and analysis.

---

# 2️⃣ Data Cleaning and Preprocessing

Data preprocessing is one of the most important stages of NLP projects.

The following preprocessing steps were performed:

* Handling missing values
* Removing invalid records
* Cleaning text data
* Lowercase conversion
* Feature engineering
* Text length extraction

Preprocessing improves the quality of textual data and enhances Machine Learning performance.

---

# 3️⃣ Exploratory Data Analysis (EDA)

EDA was performed to understand the structure and characteristics of the dataset.

The following analyses were performed:

* Toxic label distribution
* Correlation analysis between labels
* Text length distribution
* Multi-label relationship analysis

EDA helps identify hidden trends and patterns in the dataset.

---

# 📊 Advanced Visualizations

The project generates advanced visualizations including:

* Label Distribution Graph
* Correlation Heatmap
* Text Length Distribution
* Model Accuracy Comparison
* Multi-label Confusion Matrices

These visualizations help understand model behavior and dataset characteristics.

---

# 4️⃣ Feature Engineering

Machine Learning algorithms cannot directly process textual data.

Therefore, text data was converted into numerical vectors using:

## TF-IDF Vectorization

TF-IDF (Term Frequency – Inverse Document Frequency) identifies important words in comments and converts text into numerical feature vectors.

Advantages of TF-IDF:

* Highlights important words
* Reduces noise
* Improves classification performance
* Efficient for NLP tasks

---

# 5️⃣ Multi-Label Classification

This project uses Multi-Label Classification because a single comment can belong to multiple categories simultaneously.

Example:

Comment:

```text
You are disgusting and stupid
```

Labels:

* toxic
* insult

The system predicts multiple labels for each input comment.

---

# 6️⃣ Machine Learning Models

The project implements and compares multiple Machine Learning algorithms.

## Implemented Algorithms

### ✅ Logistic Regression

A powerful linear classification algorithm widely used for NLP tasks.

### ✅ Naive Bayes

A probabilistic classifier highly efficient for text classification.

### ✅ Linear SVM (Support Vector Machine)

A high-performance classification algorithm effective for sparse text features.

### ✅ Random Forest

An ensemble learning algorithm using multiple decision trees.

---

# 7️⃣ Automatic Best Model Selection

The system automatically:

* Trains all models
* Evaluates model performance
* Compares metrics
* Selects the best-performing model
* Saves the best model

This creates an intelligent model selection pipeline.

---

# 📈 Evaluation Metrics

The models were evaluated using advanced evaluation metrics including:

* Accuracy Score
* Hamming Loss
* F1 Score
* Precision
* Recall
* Classification Report
* Multi-label Confusion Matrix

These metrics help evaluate classification quality and model reliability.

---

# 🏆 Best Model

After comparing multiple algorithms, the best-performing model was automatically selected and saved for deployment.

The selected model achieved high prediction accuracy and strong multi-label classification performance.

---

# 🌐 Streamlit Web Dashboard

An interactive dashboard was developed using Streamlit to provide a professional user interface.

The dashboard allows users to:

* Enter comments
* Predict toxicity labels
* View advanced visualizations
* Analyze model performance
* Interact with the NLP system

---

# 💻 Dashboard Features

The Streamlit dashboard includes:

* Real-time prediction system
* Sidebar navigation
* Visualization viewer
* Multi-label output
* Professional UI
* Interactive experience

The dashboard converts the project into a deployable AI web application.

---

# 📂 Project Structure

```text
Advanced_Toxic_Comment_Classification/
│
├── advanced_train_model.py
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── best_toxic_model.pkl
│
├── visualizations/
│   ├── label_distribution.png
│   ├── heatmap.png
│   ├── model_comparison.png
│   ├── text_length_distribution.png
│   └── confusion_matrix_*.png
```

---

# ▶️ Installation Steps

## Step 1 — Install Python

Download Python from:

https://www.python.org/downloads/

---

## Step 2 — Install Required Libraries

Open terminal and run:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Step 1 — Train Model

Run:

```bash
python advanced_train_model.py
```

This step:

* Downloads dataset
* Trains multiple ML models
* Compares models
* Generates visualizations
* Saves best model

---

## Step 2 — Run Streamlit Dashboard

Run:

```bash
python -m streamlit run app.py
```

---

# 🌍 Open Dashboard

After successful execution open:

```text
http://localhost:8501
```

in your browser.

---

# 📌 Sample Predictions

| Input Comment                 | Predicted Labels         |
| ----------------------------- | ------------------------ |
| You are disgusting and stupid | toxic, insult            |
| I will destroy you            | threat, toxic            |
| This is a beautiful message   | No toxic labels detected |

---

# 💡 Advantages of the Project

The project provides several advantages:

* Real-world NLP implementation
* Multi-label classification system
* Automatic toxic content detection
* Advanced Machine Learning workflow
* Interactive dashboard deployment
* Professional portfolio project
* Industry-level AI application

---

# ⚠️ Limitations of the Project

Although the system performs well, some limitations include:

* Prediction quality depends on dataset quality
* Contextual understanding limitations
* Sarcasm detection challenges
* Computational requirements for large datasets

Future improvements can further improve system intelligence.

---

# 🚀 Future Enhancements

Possible future improvements include:

* Deep Learning models
* BERT Transformer integration
* LSTM networks
* Real-time API deployment
* Multi-language support
* Cloud deployment
* AI moderation integration
* Explainable AI techniques

---

# 🧠 Learning Outcomes

This project helps understand:

* Natural Language Processing
* Multi-label classification
* Text preprocessing
* TF-IDF vectorization
* Machine Learning pipelines
* Model evaluation
* Streamlit deployment
* Data visualization
* End-to-end AI workflows

---

# 📌 Real-World Applications

This system can be used in:

* Social media moderation
* YouTube comment filtering
* Gaming chat moderation
* AI safety systems
* Community moderation tools
* Online discussion platforms
* Content filtering systems

---

# 📚 Conclusion

The Advanced Multi-Label Toxic Comment Classification System successfully demonstrates how Natural Language Processing and Machine Learning can be used to automatically detect harmful online comments.

The project integrates:

* Data collection
* Text preprocessing
* Feature engineering
* Multi-label classification
* Model comparison
* Advanced visualizations
* Web deployment

into a complete intelligent AI application.

This project provides strong practical experience in NLP, Machine Learning, data visualization, and AI deployment, making it an excellent industry-level internship project and portfolio project.





































