# **LSTM Text Classification Model Using Pytorch**

<img width="1736" height="1157" alt="ProjectArchitecture" src="https://github.com/user-attachments/assets/03387367-15bf-4de6-b7a4-69de78ac30e6" />

This project implements a **Long Short-Term Memory (LSTM)** model for classifying app reviews on a scale of **1 to 5**. The model is built using **PyTorch** and aims to predict the sentiment of the reviews based on the provided text. The dataset contains app review texts and their corresponding ratings.

---

## **Project Overview**

LSTM (Long Short-Term Memory) is a special type of **Recurrent Neural Network (RNN)** designed to address the limitations of vanilla RNNs, especially with regard to learning long-term dependencies. LSTMs are particularly effective in applications such as **text classification**, **speech recognition**, and **time series forecasting**.

In this project, you will learn how to:

- **Understand LSTM working**.
- **Classify app reviews** based on user feedback (ratings from 1 to 5).



---

## **Tech Stack**

- **Language**: Python
- **Libraries**:
  - **pandas**: Data manipulation and analysis.
  - **tensorflow**: For building the model (though the primary model is in PyTorch, tensorflow is used for tokenization and other pre-processing).
  - **matplotlib**: For visualizing the results (training & test losses).
  - **scikit-learn**: For data preprocessing and splitting the dataset.
  - **nltk**: For natural language processing tasks (tokenization, stopword removal, lemmatization).
  - **numpy**: For numerical operations.
  - **PyTorch**: For building and training the LSTM model.

---

## **Data Description**

The dataset contains:

- **Content**: Text of the review.
- **Score**: Rating score (between 1 and 5).

The **score** column is the target variable, and **content** is the feature used for prediction. The model learns to predict the review score (1-5) based on the review text.

---

## **Project Structure**

Here’s the directory structure of the project:

```
LSTM-Text-Classification/
│
├── data/
│   ├── review_data.csv            # Dataset containing review texts and ratings
│
├── MLPipeline/
│   ├── __init__.py                # Initialization file for MLPipeline
│   ├── Create.py                  # Contains functions to split dataset and create data loaders
│   ├── Load_Data.py               # Loads the dataset from CSV
│   ├── Preprocessing.py           # Data cleaning and preprocessing functions
│   ├── Tokenisation.py            # Tokenization methods and text vectorization
│   ├── Lstm.py                    # LSTM model architecture
│   └── Train_Test.py              # Training and evaluation functions
│
├── notebooks/
│   ├── LSTM_Text_Classification.ipynb   # Jupyter Notebook for building the model
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation (this file)
```

---

## **Approach**

### **1. Data Preprocessing**

- **Read Data**: The data is read from a CSV file that contains **review text** and **score** (ratings from 1 to 5).
- **Drop Unwanted Features**: Any irrelevant columns or features are dropped to keep only the essential data.
- **Text Cleaning**:
  - Lowercase the text.
  - Remove punctuation and links.
  - Remove digits and unwanted characters.
- **Handle Class Imbalance**: The dataset contains imbalanced classes (more reviews with high ratings). We use upsampling for minority classes and downsampling for the majority class to balance the data.
- **Tokenization of Text**: Convert the text into tokens (numeric representation of words) using **Keras Tokenizer**.
- **Label Encoding**: Convert the ratings into numeric labels (1-5) for model training.

### **2. Train/Test Split**

- The dataset is split into **training** and **testing** sets using `train_test_split` from **scikit-learn**.

### **3. Convert Data to Tensors**

- Convert tokenized text data into **PyTorch tensors** for use in the model.

### **4. LSTM Model Building**

- **LSTM Model**: The core of the model is an LSTM layer with an embedding layer to process the sequences of tokens.
- **Architecture**: The LSTM network is designed with:
  - An embedding layer for vector representation of words.
  - An LSTM layer to capture the sequence dependencies.
  - A fully connected output layer for classification (predicting the review score).

### **5. Model Training**

- The model is trained using **PyTorch**. We compile the model, define the loss function (`CrossEntropyLoss`), and use an optimizer (`Adam`).

### **6. Model Evaluation**

- Evaluate the model's performance on the **test data** and calculate metrics like **accuracy**.

### **7. Visualizing Results**

- **Losses**: Track and visualize the **training and testing losses** over epochs to assess model convergence.

---

## **Model Evaluation**

After training, the model is evaluated on the test data. Key metrics include:

- **Accuracy**: Measures the percentage of correct predictions.
- **Loss**: Tracks how well the model is fitting the data. Lower loss indicates a better fit.
- **Confusion Matrix**: Used for multi-class classification to understand true positives, false positives, and other evaluation aspects.

---


## **Conclusion**

This project provides a hands-on guide to building an **LSTM-based text classification model** for classifying app reviews. By leveraging **PyTorch** and **natural language processing techniques**, the model can learn from textual data and predict app ratings based on user feedback. The final model can be further enhanced and deployed for real-time use cases, such as sentiment analysis or review-based recommendation systems.

---

## **Future Improvements**

- Implement **Bidirectional LSTMs** to capture both past and future dependencies in text.
- Experiment with **attention mechanisms** to improve model performance on long sequences.
- Explore **transformers** (e.g., BERT) for better handling of textual data.
