# JobShield — AI-Powered Fake Job Posting Detector

JobShield is a machine learning project that detects whether a job posting is **Real** or **Fake**.

The project uses text information from job postings along with categorical and binary features to train a classification model.

## Features

* Detects fake job postings using machine learning
* Uses TF-IDF for text features
* Uses OneHotEncoder for categorical features
* Uses Linear SVM for classification
* FastAPI backend for making predictions
* Simple API endpoint for job prediction

## Dataset

The project uses the **Real or Fake Job Posting Prediction** dataset from Hugging Face.

The dataset contains job posting information such as:

* Job title
* Company profile
* Description
* Requirements
* Benefits
* Location
* Employment type
* Required experience
* Required education
* Industry
* Function
* Telecommuting
* Company logo
* Questions

The target column is:

`fraudulent`

where:

* `0` = Real job
* `1` = Fake job

## Machine Learning

Different classification models were tested:

* Logistic Regression
* Naive Bayes
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Linear SVM

The final model selected was **Linear SVM**.

### Final Model Performance

| Metric             |  Score |
| ------------------ | -----: |
| Accuracy           | 99.05% |
| Fake Job Precision |    98% |
| Fake Job Recall    |    82% |
| Fake Job F1-Score  |    89% |

Fake-job recall and F1-score were given more importance because detecting fake job postings is the main objective of the project.

## Preprocessing

Different types of features are processed separately:

```text
Text Features
      ↓
    TF-IDF

Categorical Features
      ↓
 OneHotEncoder

Binary Features
      ↓
   Passthrough

      ↓
 Combined Features
      ↓
   Linear SVM
```

The preprocessing and model are stored together inside a Scikit-learn Pipeline.

## Project Structure

```text
JobShield/
│
├── src/
│   ├── app.py
│   └── jobshield_model.pkl
│
├── results/
│   └── svm_results.csv
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

## Running the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd JobShield
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the FastAPI server

```bash
uvicorn src.app:app --reload
```

The API will run locally at:

```text
http://127.0.0.1:8000
```

## API

### `GET /`

Checks whether the API is running.

Example response:

```json
{
  "message": "JobShield API is running"
}
```

### `POST /predict`

Accepts job posting information and returns whether the posting is **Real** or **Fake**.

Example response:

```json
{
  "prediction": "Fake"
}
```

## Future Improvements

* Build a frontend interface for submitting job postings
* Improve fake-job recall through further model experimentation
* Add confidence/probability information where appropriate
* Deploy the application
* Improve input validation and API security

## Disclaimer

JobShield is an educational machine learning project. Its predictions should not be treated as definitive proof that a job posting is fraudulent.
