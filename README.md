# Real-Time Spam Detection Using Bayes' Theorem

## Project Overview
This project implements a real-time spam detection system using Naive Bayes' Theorem and Natural Language Processing (NLP) techniques. It classifies messages as either "Spam" or "Ham" (not spam) using a Multinomial Naive Bayes classifier. The project provides a user-friendly GUI built with Tkinter for loading datasets, training the model, visualizing results, and classifying messages.

---

## Features
- **Dataset Loading**: Load CSV files with `Category` and `Message` columns.
- **Text Preprocessing**: Tokenization, stopword removal, and feature extraction.
- **Model Training**: Train a Naive Bayes classifier with real-time accuracy calculation.
- **Message Classification**: Classify user-input messages as spam or ham.
- **Spam/Ham Count Visualization**: View bar charts showing spam vs. ham counts.
- **Graphical User Interface**: Intuitive GUI for easy interaction.

---

## Installation
### Prerequisites:
Ensure the following are installed on your system:
- Python (3.7 or higher)
- pip (Python package manager)

### Required Libraries:
Install the required Python libraries using the following command:
```bash
pip install pandas nltk scikit-learn matplotlib
```

---

## How to Run

1. Run the script:
```bash
python main.py
```
2. Use the GUI to load a dataset, train the model, classify messages, or visualize spam/ham counts.

---

## Usage Instructions
### Loading a Dataset:
- Use the "Load Dataset" button to select a CSV file.
- The dataset must contain `Category` (spam or ham) and `Message` columns.

### Training the Model:
- After loading the dataset, click "Train Classifier" to preprocess the data and train the Naive Bayes classifier.
- The system will display the training accuracy upon successful training.

### Classifying Messages:
- Enter a message in the text box.
- Click "Classify Message" to classify the input as spam or ham.

### Viewing Spam/Ham Counts:
- Click "Show Spam/Ham Counts" to display a bar chart showing the distribution of spam and ham messages in the dataset.

---

## Screenshots
### Main GUI Interface:
![image](https://github.com/user-attachments/assets/10c24c7b-3635-4268-8e73-a757ce02cc17)

### Classifier trained accuracy
![image](https://github.com/user-attachments/assets/c7ae85f1-1888-47a9-9f5a-17def43b0784)

### Spam vs Ham Count Visualization:
![image](https://github.com/user-attachments/assets/cad6584b-23e5-4124-a8d2-adf817bdd49d)

### Message Classifications as
**Spam**

![image](https://github.com/user-attachments/assets/807baf5b-cf90-4c9f-9359-60182da4862f)
![image](https://github.com/user-attachments/assets/566c0e97-bce8-471b-9608-ee12ef4123e0)

**Ham**

![image](https://github.com/user-attachments/assets/30fc3cc4-f13c-49c2-a1fe-4525df29c5ba)
![image](https://github.com/user-attachments/assets/62202076-b0f5-498b-a5f0-56063a2477e8)

---

## License
This project is licensed under the [MIT License](LICENSE).
