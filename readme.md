# ❤️ Heart Disease Detection System

## Overview

A full-stack Machine Learning web application that predicts the
likelihood of heart disease based on patient clinical data. The project
combines data processing, model training, and a FastAPI-based web
interface, and is designed to be production-ready.

------------------------------------------------------------------------

## 🎯 Project Objectives

-   Build a machine learning model to predict heart disease
-   Validate user inputs using Pydantic
-   Serve predictions through a FastAPI backend
-   Provide a clean HTML-based frontend
-   Prepare the project for Dockerization and AWS deployment

------------------------------------------------------------------------

## 🧠 What Was Done

### Data Pipeline

-   Raw data ingestion and schema validation
-   Data preprocessing including scaling and label encoding

### Model Training

-   Logistic Regression model trained on processed data
-   Model evaluation performed before saving artifacts

### Backend

-   FastAPI application to serve predictions
-   Pydantic models for strict input validation
-   Jinja2 templates for rendering HTML pages

### Frontend

-   Clean and modern HTML/CSS UI
-   Radio buttons and numeric inputs for better UX
-   Prediction results rendered dynamically

------------------------------------------------------------------------

## 🗂 Project Structure

-   artifacts/ → Trained model, scaler, encoders
-   data/ → Raw and processed datasets
-   src/heart_disease → Application source code
-   templates/ → HTML templates
-   logs/ → Application logs

------------------------------------------------------------------------

## 🚀 Deployment

-   Docker-ready project structure
-   Intended deployment on AWS using Docker containers

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python
-   FastAPI
-   Pydantic
-   Scikit-learn
-   HTML / CSS
-   Docker
-   AWS

------------------------------------------------------------------------

## ✅ Status

The project is complete and ready for containerization and cloud
deployment.
