# Documentation for Resume Categorization using Machine Learning

This repository contains code and resources for performing resume categorization using classification techniques. The goal is to automatically classify resumes into different categories based on their contents.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)

## Introduction

The task of resume categorization is important for various applications, such as job portals, recruitment agencies, and HR departments. This project explores the use of machine learning algorithms for automatically categorizing resumes into 24 different predefined categories, such as "Teacher," "Designer," and "Engineering" etc.

## Installation

To use the code in this repository, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Nehlr1/Resume-Categorization-with-Classification.git

2. Clone the repository:

   ```bash
   pip install -r requirements.txt

3. Run the resume_classification.ipynb file to create the model or use the models from the "model" folder.

4. Run the script.py
    
   ```bash
   python script.py filepath/

> **Note**
> Here filepath/ indicates the folder where the resumes, in pdf format, are located to be reorganised into its respective folder 


## Usage
To use the code in this repository, follow these steps:

1. Import necessary Libraries and read the "Resume.csv" dataset.

2. Data Exploration: This step involves performing exploratory data analysis (EDA) to understand data distribution and identifying patterns that could affect model performance.

3. Preprocess the resume data: This step involves cleaning the text data, removing stopwords, tokenization, stemming, lemmatization etc. And encoding the "Category" Column, splitting the dataset and finally, creating TF-IDF vectors for training and testing data, using 800 features and English stop words.

4. Train the classification model: Various classification algorithms were used, such as Random Forest Classifier, Logistic Regression, Light GBM Classifier and XGBoost Classifier. Experimented with different algorithms and hyperparameters to find the best model for the dataset.

4. Evaluation of the model: Used suitable evaluation metric, such as accuracy, precision, recall, or F1-score, to assess the performance of the trained model. Light GBM Classifier had the best results.

5. Trained Model: Use the trained model (Light GBM Classifier Model) for resume categorization. Once the model is trained and evaluated, it was used to predict the category of new resumes.

## File Structure
The file structure of this repository is as follows:

1. classification_resume.ipynb: Jupyter Notebook containing the code for resume categorization. It provides a step-by-step guide and explanations of the code.

2. script.py: Python script version of the code for resume categorization.

3. model/: Directory containing saved models.

4. categorized_resumes.csv: The CSV have two columns named filename and category which is created using script.py.
