# SMS Spam Detection using Naive Bayes

## Overview
This project implements an SMS spam detection system using the Naive Bayes algorithm. The goal is to classify SMS messages as either "Spam" or "Ham" (not spam). The project is developed in Python and demonstrated in a Jupyter Notebook.

## Features
- Preprocessing of SMS text data, including tokenization and text cleaning.
- Implementation of the Naive Bayes algorithm for text classification.
- Evaluation of model performance using standard metrics like accuracy, precision, recall, and F1-score.

## Dataset
The dataset used in this project contains SMS messages labeled as "spam" or "ham." It is a publicly available dataset widely used for spam detection tasks. The dataset should be placed in the same directory as the notebook before running the project.

## Requirements
The following Python libraries are required to run this project:
- pandas
- numpy
- scikit-learn
- nltk
- Jupyter Notebook

You can install the dependencies using:
```bash
pip install pandas numpy scikit-learn nltk
```

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/vikastomar0010/sms-spam-detection.git
```
2. Navigate to the project directory:
```bash
cd sms-spam-detection
```
3. Open the Jupyter Notebook:
```bash
jupyter notebook "sms_spam_detection using with naive bayes.ipynb"
```
4. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

## Project Structure
- `sms_spam_detection using with naive bayes.ipynb`: Main Jupyter Notebook containing the implementation.
- `data/`: Directory to store the dataset (not included; download and place the dataset here).
- `README.md`: Project documentation.

## Results
The model achieved high accuracy in detecting spam messages, demonstrating the effectiveness of the Naive Bayes algorithm for this task. Detailed evaluation results can be found in the notebook.

## Future Work
- Experiment with other machine learning algorithms like Support Vector Machines and Logistic Regression.
- Use advanced text representations like TF-IDF or word embeddings.
- Build a web interface to deploy the spam detection system.

## Acknowledgments
- Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- Tutorials and documentation from [scikit-learn](https://scikit-learn.org/stable/) and [NLTK](https://www.nltk.org/).


