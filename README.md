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
cv_parse/
├── training.py              # Model training pipeline
├── validate.py              # Model validation
├── db_cv.json              # Training dataset (CV data)
├── db_merged.json          # Merged dataset
├── db_notcv.json           # Non-CV data
├── output/                 # Trained model artifacts
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── config.json
└── server_files/           # Flask web application
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

Please note the db.json file is an example, and isn't what we trained the model on.

### Validating the Model

```bash
python validate.py
```

### Running the Web Server

```bash
cd server_files
python mainfile.py
```

Access the application at `http://localhost:5000`

### Using Inference Directly

```python
from server_files.inference import get_tags

entities = get_tags('path/to/cv.pdf')
print(entities)  # Returns dict of entities by label
```

## Model Details

- **Base Model**: camembert-base (French BERT)
- **Task**: Token Classification (NER)
- **Framework**: Hugging Face Transformers
- **Training Device**: GPU (if available) / CPU

## Output Structure

The inference returns a dictionary with entity labels as keys:

```python
{
    "name": ["John Doe"],
    "skills": ["Python", "React", "SQL"],
    "mail": ["john@example.com"],
    "location": ["Paris, France"],
    ...
}
```

## Deployment

The project includes WSGI configuration (server_files/wsgi.py) for production deployment with Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 mainfile:app
```