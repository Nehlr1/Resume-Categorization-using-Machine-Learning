import os
import argparse
import csv
import joblib
import PyPDF2
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import download
download("punkt")
download("stopwords")
download("wordnet")


def resume_cleaning(text):
    # Removing HTML tags 
    cleaned_text = re.sub(r'<.*?>', ' ', text)
    
    # Removing URLs
    cleaned_text = re.sub(r'http\S+', ' ', cleaned_text)
    
    # Removing non-alphabetical characters, punctuation, special characters, digits, continuous underscores, and extra whitespace
    cleaned_text = re.sub('[^a-zA-Z]', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Converting to lowercase
    cleaned_text = cleaned_text.lower()
    
    # Tokenizing the cleaned text
    words = word_tokenize(cleaned_text)
    
    # Removing stopwords
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Applying stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    # Applying lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]
    
    # Joining words back into a single string
    cleaned_text = " ".join(lemmatized_words)
    
    return cleaned_text

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def categorize_resume(resume, model, vectorizer, encoder):
    cleaned_resume = resume_cleaning(resume)  # Cleaning the resume text
    features = vectorizer.transform([cleaned_resume])  # Vectorizing the cleaned text
    encoded_category = model.predict(features)[0]  # Predicting the encoded category
    return encoder.inverse_transform([encoded_category])[0]  # Decoding to original category name

def process_directory(dir_path, model, vectorizer, encoder):
    output_csv = os.path.join(dir_path, "categorized_resumes.csv")

    with open(output_csv, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["filename", "category"])

        for filename in os.listdir(dir_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(dir_path, filename)
                content = extract_text_from_pdf(file_path)
                category = categorize_resume(content, model, vectorizer, encoder)
                csvwriter.writerow([filename, category])

                # Using the category name for creating the directory
                dest_dir = os.path.join(dir_path, category)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                os.rename(file_path, os.path.join(dest_dir, filename))
    
    # Printing confirmation only if process_directory is executed
    print("Resume Processing Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and categorize resumes")
    parser.add_argument("dir_path", type=str, help="Path to the directory containing resumes")
    args = parser.parse_args()

    # Loading the pre-trained model, vectorizer, and label encoder
    model = joblib.load("model/light_bgm_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    encoder = joblib.load("model/label_encoder.pkl")

    # Processing the directory containing resumes
    process_directory(args.dir_path, model, vectorizer, encoder)