import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

# Global variables
data = None
vectorizer = None
model = None
X_train, X_test, y_train, y_test = None, None, None, None


# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())  # Tokenize and lowercase
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)


# Function to load dataset
def load_dataset():
    global data
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        data = pd.read_csv(file_path)

        # Check if 'Category' and 'Message' columns exist
        if 'Category' not in data.columns or 'Message' not in data.columns:
            messagebox.showerror("Error", "Dataset must contain 'Category' and 'Message' columns!")
            data = None
            return

        messagebox.showinfo("Success", "Dataset loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")


# Function to train the classifier
def train_classifier():
    global data, model, vectorizer, X_train, X_test, y_train, y_test
    if data is None:
        messagebox.showerror("Error", "Dataset not loaded!")
        return

    try:
        # Ensure all values in the 'Message' column are strings and handle missing values
        data['Message'] = data['Message'].fillna("").astype(str)

        # Remove rows with missing labels
        data = data[data['Category'].notna()]
        data['Category'] = data['Category'].astype(str)

        # Preprocess text
        data['processed_text'] = data['Message'].apply(preprocess_text)

        # Feature extraction
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data['processed_text'])

        # Map labels: 'ham' -> 0, 'spam' -> 1
        valid_labels = {'ham': 0, 'spam': 1}
        data['Category'] = data['Category'].map(valid_labels)
        data = data[data['Category'].notna()]  # Remove invalid labels
        y = data['Category']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Naive Bayes classifier
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Calculate accuracy on the test set
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # Show success message
        messagebox.showinfo("Success", f"Classifier trained successfully!\nAccuracy: {acc:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train classifier: {str(e)}")


# Function to classify a new message
def classify_message():
    global model, vectorizer
    if model is None or vectorizer is None:
        messagebox.showerror("Error", "Model not trained!")
        return

    try:
        message = input_message.get("1.0", tk.END).strip()
        if not message:
            messagebox.showerror("Error", "Message cannot be empty!")
            return

        processed_message = preprocess_text(message)
        features = vectorizer.transform([processed_message])
        prediction = model.predict(features)

        result = "Spam" if prediction[0] == 1 else "Ham"
        messagebox.showinfo("Result", f"The message is classified as: {result}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to classify message: {str(e)}")


# Function to display spam and ham count
def show_counts():
    global data
    if data is None:
        messagebox.showerror("Error", "Dataset not loaded!")
        return

    try:
        # Calculate counts
        spam_count = sum(data['Category'] == 1)
        ham_count = sum(data['Category'] == 0)

        # Display counts in a bar chart
        plt.bar(['Ham', 'Spam'], [ham_count, spam_count], color=['blue', 'red'])
        plt.title("Spam vs Ham Count")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to display counts: {str(e)}")


# Function to exit the application
def exit_app():
    root.destroy()


# Create the GUI
root = tk.Tk()
root.title("Spam Detection with Naive Bayes")

# Buttons
btn_load = tk.Button(root, text="Load Dataset", command=load_dataset, width=25)
btn_load.pack(pady=5)

btn_train = tk.Button(root, text="Train Classifier", command=train_classifier, width=25)
btn_train.pack(pady=5)

btn_counts = tk.Button(root, text="Show Spam/Ham Counts", command=show_counts, width=25)
btn_counts.pack(pady=5)

label_input = tk.Label(root, text="Enter a message to classify:")
label_input.pack(pady=5)

input_message = tk.Text(root, height=5, width=50)
input_message.pack(pady=5)

btn_classify = tk.Button(root, text="Classify Message", command=classify_message, width=25)
btn_classify.pack(pady=5)

btn_exit = tk.Button(root, text="Exit", command=exit_app, width=25)
btn_exit.pack(pady=5)

root.mainloop()
