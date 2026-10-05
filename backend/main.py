from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI(
    title="JobShield API",
    description="Fake job posting detection API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load("src/jobshield_model.pkl")


class JobInput(BaseModel):
    title: str
    company_profile: str
    description: str
    requirements: str
    benefits: str
    location: str
    employment_type: str
    required_experience: str
    required_education: str
    industry: str
    function: str
    telecommuting: int
    has_company_logo: int
    has_questions: int


@app.get("/")
def home():
    return {
        "message": "JobShield API is running"
    }


@app.post("/predict")
def predict_job(job: JobInput):

    input_data = pd.DataFrame([{
        "title": job.title,
        "company_profile": job.company_profile,
        "description": job.description,
        "requirements": job.requirements,
        "benefits": job.benefits,
        "location": job.location,
        "employment_type": job.employment_type,
        "required_experience": job.required_experience,
        "required_education": job.required_education,
        "industry": job.industry,
        "function": job.function,
        "telecommuting": job.telecommuting,
        "has_company_logo": job.has_company_logo,
        "has_questions": job.has_questions
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Fake"
    else:
        result = "Real"

    return {
        "prediction": result
    }