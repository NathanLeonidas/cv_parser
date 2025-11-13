A machine learning-powered CV parsing system that extracts structured information from resume PDFs using Named Entity Recognition (NER). This project was made for Sysnav, a French company proposing machine-learning assisted navigation solutions.

![image](./server_files/logo.png)

## Overview

This project uses a fine-tuned **CamemBERT** model to identify and extract key information from CVs, including:
- Personal information (name, contact details, location)
- Professional skills
- Work experience
- Academic background
- Languages and interests

Sysnav required a lightweight, scalable, non-hallucinating model that could precisely categorize applicants.

## Project Structure

```

training.py              # Model training pipeline
validate.py              # Model validation
db_cv.json              # Training dataset (CV data)
db_merged.json          # Merged dataset
db_notcv.json           # Non-CV data
output/                 # Trained model artifacts
├── model.safetensors
├── tokenizer.json
└── config.json
server_files/           # Flask web application
├── mainfile.py         # Flask app & routes
├── inference.py        # CV parsing inference
├── read_pdf.py         # PDF processing
├── wsgi.py             # WSGI configuration
├── templates/          # HTML templates
└── uploads/            # Uploaded CV storage
```

## Components

### Training & Model (training.py)
- Fine-tunes CamemBERT for token classification
- Processes annotated CV data with entity labels (BIO tagging)
- Uses 80/20 train/test split
- Trains for 100 epochs with learning rate 2e-4

### Inference (server_files/inference.py)
- Loads the trained model and tokenizer
- Extracts text from PDF CVs
- Performs entity extraction using BIO tag predictions
- Returns entities grouped by label

### Web Server (server_files/mainfile.py)
- Flask application with PDF upload interface
- Routes:
  - `/` - Home page
  - `/upload` - Single file upload
  - `/uploads` - Multiple file uploads
- Validates PDF files (max 20MB)

### PDF Processing (server_files/read_pdf.py)
- Extracts text from PDF documents
- Text preprocessing and normalization

## Entity Labels

The model recognizes the following entity types:
- `name` - Candidate name
- `location` - Geographic locations
- `telephone` - Phone numbers
- `mail` - Email addresses
- `languages` - Languages spoken
- `interests` - Personal interests
- `skills` - Technical and soft skills
- `entreprises` - Company names
- `experience` - Work experience
- `academics` - Education and qualifications

## Installation

```bash
# Create virtual environment
python -m venv serv_env
source serv_env/bin/activate  # On Windows: serv_env\Scripts\activate

# Install dependencies
pip install transformers torch datasets evaluate flask gunicorn PyPDF2
```

## Usage

### Training the Model

```bash
python training.py
```
We only provided a mock database db.json, even though weights of the model in /output were trained on synthetic data.

### Validating the Model on given Text

```bash
python validate.py
```

### Running the Web Server
Only a sample of the full (now retired) web server has been backed up, the rest is at SYSNAV.
Running this (thus incomplete) web server will require you to adjust the path used to save the pdfs in mainfile.py and the path used to deserialize the model in inference.
Access the application at `http://localhost:5000`
Deployment on remote server requires a running a gunicorn and nginx daemon (use the provided wsgi.py if needed).






