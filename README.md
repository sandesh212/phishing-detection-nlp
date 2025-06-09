# Phishing Detection Using NLP

This project implements a **Phishing Email Detection System** using Natural Language Processing (NLP) techniques. It analyzes email content to classify emails as **Phishing** (fraudulent emails attempting to steal sensitive information) or **Legitimate** (genuine emails). The system leverages a combination of traditional machine learning models, a feedforward neural network, and a Long Short-Term Memory (LSTM) model to achieve high accuracy in detecting phishing attempts. It also provides a prediction function to classify new email texts.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [How to Predict](#how-to-predict)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
Phishing emails pose a significant cybersecurity threat by pretending to be from trustworthy sources to steal sensitive information like passwords or credit card details. This project uses NLP to process email text, extract meaningful features, and train multiple models to classify emails as phishing (label: 1) or legitimate (label: 0). The project includes data preprocessing, feature extraction, model training, evaluation, and a prediction function for real-time email classification.

## Features
- **Data Preprocessing**: Cleans email text by converting to lowercase, removing URLs, punctuation, and extra spaces, and applies tokenization, stop word removal, and lemmatization.
- **Feature Extraction**: Uses TF-IDF vectorization for traditional machine learning models and tokenization/padding for deep learning models.
- **Multiple Models**: Implements eight traditional machine learning models and two deep learning models (Neural Network and LSTM).
- **Model Evaluation**: Evaluates models using accuracy, precision, recall, and F1-score, with a bar plot to compare traditional model performance.
- **Prediction Function**: Provides a function to classify new email text as phishing or legitimate using the trained LSTM model.

## Dataset
The project uses the **CEAS_08 dataset** (`CEAS_08.csv`), which contains email data with the following columns:
- `sender`: Email sender address.
- `receiver`: Email recipient address.
- `date`: Date and time the email was sent.
- `subject`: Email subject line.
- `body`: Email content.
- `label`: Binary label (0 for legitimate, 1 for phishing).
- `urls`: Number of URLs in the email (not extensively used in the code).

**Note**: The dataset is not included in the repository due to size constraints. You must obtain the `CEAS_08.csv` file separately and place it in the project directory. Alternatively, you can modify the code to use another email dataset with similar structure.

## Requirements
To run this project, you need the following:
- **Python**: Version 3.12.3
- **Libraries**:
  - `pandas` (data manipulation)
  - `numpy` (numerical operations)
  - `nltk` (NLP processing)
  - `matplotlib` (visualization)
  - `seaborn` (visualization)
  - `scikit-learn` (machine learning models and metrics)
  - `tensorflow` (deep learning models)
  - `keras` (deep learning models)
  - `xgboost` (XGBoost model)
  - `lightgbm` (LightGBM model)
  - `transformers` (optional, not used in the current implementation)

## Installation
Follow these steps to set up the project:
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sandesh212/phishing-detection-nlp.git
   cd phishing-detection-nlp
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure `requirements.txt` is in the project directory, then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**:
   Run the following Python code to download required NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

5. **Place the Dataset**:
   - Obtain the `CEAS_08.csv` file and place it in the project directory.
   - Ensure the notebook uses a relative path (e.g., `pd.read_csv("CEAS_08.csv")`).

## Usage
1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook PhishingDetectionUsingNLP.ipynb
   ```

2. **Run the Notebook**:
   - Execute the cells sequentially to:
     - Load and preprocess the dataset.
     - Train and evaluate traditional machine learning and deep learning models.
     - Visualize model performance using a bar plot.
     - Test the LSTM-based prediction function on new email texts.

3. **Example Predictions**:
   Use the `predict_lstm` function to classify new emails:
   ```python
   # Phishing email
   print(predict_lstm("You have won $10,000! Click here to claim."))
   # Output: Phishing

   # Legitimate email
   print(predict_lstm("Hi Carla, How are you? Everything is fine here, just wanted to let you know My daughter graduated last week."))
   # Output: Legitimate

   # Phishing email
   print(predict_lstm("Once you click the following link, you will get a reward of 1000$ for the first time."))
   # Output: Phishing
   ```

## Project Structure
```
phishing-detection-nlp/
│
├── PhishingDetectionUsingNLP.ipynb  # Main Jupyter Notebook with the implementation
├── CEAS_08.csv                    # Dataset (not included, user must provide)
├── README.md                       # This file
├── requirements.txt                # List of dependencies
├── .gitignore                     # Git ignore file for excluding unnecessary files
```

## Methodology
The project follows these steps:
1. **Data Loading**:
   - Loads the CEAS_08 dataset using Pandas.
2. **Data Preprocessing**:
   - **Cleaning**: Converts text to lowercase, removes URLs, punctuation, and extra spaces.
   - **Tokenization**: Splits text into words using NLTK.
   - **Stop Word Removal**: Removes common words (e.g., "the", "is") that add little meaning.
   - **Lemmatization**: Reduces words to their base form (e.g., "running" to "run").
3. **Feature Extraction**:
   - For traditional ML: Uses TF-IDF vectorization (5,000 features).
   - For deep learning: Uses tokenization and padding to create fixed-length sequences (10,000 vocab size, 100-token length).
4. **Train-Test Split**:
   - Splits data into 80% training and 20% testing sets.
5. **Model Training**:
   - Trains multiple traditional ML and deep learning models.
6. **Evaluation**:
   - Measures accuracy, precision, recall, and F1-score.
   - Visualizes traditional ML model performance with a bar plot.
7. **Prediction**:
   - Provides a function to classify new emails using the LSTM model.

## Models
The project implements the following models:
1. **Traditional Machine Learning Models**:
   - **Multinomial Naive Bayes**: Probabilistic model for text classification.
   - **Logistic Regression**: Linear model with 1,000 max iterations.
   - **Random Forest**: Ensemble of 100 decision trees.
   - **Decision Tree**: Single decision tree classifier.
   - **Gradient Boosting**: Sequential ensemble of decision trees.
   - **Linear SVM**: Support Vector Machine with a linear kernel.
   - **XGBoost**: Optimized gradient boosting model.
   - **LightGBM**: Fast and efficient gradient boosting model.
2. **Deep Learning Models**:
   - **Feedforward Neural Network**:
     - Layers: 128 (ReLU) → Dropout (0.3) → 64 (ReLU) → 1 (sigmoid).
     - Trained for 5 epochs with a batch size of 32.
   - **LSTM Model**:
     - Layers: Embedding (10,000 vocab, 128 dimensions) → LSTM (64 units) → Dropout (0.5) → Dense (32, ReLU) → Dense (1, sigmoid).
     - Trained for 5 epochs with a batch size of 32.

## Results
- **Traditional Machine Learning Models**:
  - Linear SVM: 0.994 accuracy
  - Logistic Regression: 0.991 accuracy
  - Random Forest: 0.990 accuracy
  - LightGBM: 0.989 accuracy
  - XGBoost: 0.988 accuracy
  - Decision Tree: 0.979 accuracy
  - Multinomial Naive Bayes: 0.972 accuracy
  - Gradient Boosting: 0.970 accuracy
- **Deep Learning Models**:
  - Feedforward Neural Network: ~0.99 accuracy
  - LSTM: ~0.993 accuracy
- **Visualization**: A bar plot compares the accuracy of traditional ML models.
- **Prediction**: The LSTM model accurately classifies new emails, as shown in the example predictions.

## How to Predict
Use the `predict_lstm` function to classify new email text:
```python
def predict_lstm(text):
    text_seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(text_seq, maxlen=100, padding='post')
    pred = model.predict(padded)[0][0]
    return 'Phishing' if pred >= 0.5 else 'Legitimate'

# Example usage
text = "You have won $10,000! Click here to claim."
result = predict_lstm(text)
print(result)  # Output: Phishing
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add new feature or fix"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a Pull Request on GitHub.

Please ensure your code follows the project’s coding style and includes appropriate documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **CEAS_08 Dataset**: For providing a comprehensive email dataset for phishing detection research.
- **Libraries**: Thanks to `pandas`, `scikit-learn`, `tensorflow`, `nltk`, `xgboost`, and `lightgbm` for their robust tools.
- **Community**: Inspired by NLP and cybersecurity research communities.

---

**Note**: If you encounter issues with the dataset path or dependencies, please check the file paths and ensure all libraries are installed correctly. For further assistance, open an issue on the GitHub repository.