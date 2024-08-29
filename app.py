from flask import Flask, request, jsonify, render_template, send_file
import os
import pdfplumber
import re
from transformers import pipeline
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fpdf import FPDF

# Initialize the Flask application
app = Flask(__name__)

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text

# Function to load and process the transcript
def load_transcript(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        raise ValueError("Unsupported file format")

# Function to perform zero-shot classification
def extract_information(text, labels):
    result = classifier(text, labels)
    return {label: result['scores'][result['labels'].index(label)] if label in result['labels'] else 0.0 for label in labels}

# Functions to extract information
def extract_customer_requirements(text):
    labels = ["Hatchback", "SUV", "Sedan", "Petrol", "Diesel", "Electric", "Red", "Blue", "White", "Black", "Manual", "Automatic"]
    return extract_information(text, labels)

def extract_company_policies(text):
    labels = ["Free RC Transfer", "5-Day Money Back Guarantee", "Free RSA for One Year", "Return Policy"]
    return extract_information(text, labels)

def extract_customer_objections(text):
    labels = ["Refurbishment Quality", "Car Issues", "Price Issues", "Customer Experience Issues"]
    return extract_information(text, labels)

# Function to structure output in JSON format
def structure_output(requirements, policies, objections):
    output = {
        "Customer Requirements": requirements,
        "Company Policies Discussed": policies,
        "Customer Objections": objections
    }
    return output

# Function to process the transcript
def process_transcript(file_path):
    transcript = preprocess_text(load_transcript(file_path))
    requirements = extract_customer_requirements(transcript)
    policies = extract_company_policies(transcript)
    objections = extract_customer_objections(transcript)
    return structure_output(requirements, policies, objections)

# Function to create visualizations
def vis(data):
    requirements_df = pd.DataFrame(columns=["Requirement", "Score"])
    objections_df = pd.DataFrame(columns=["Objection", "Score"])

    for entry in data:
        req_df = pd.DataFrame(list(entry["Customer Requirements"].items()), columns=["Requirement", "Score"])
        obj_df = pd.DataFrame(list(entry["Customer Objections"].items()), columns=["Objection", "Score"])
        requirements_df = pd.concat([requirements_df, req_df])
        objections_df = pd.concat([objections_df, obj_df])

    requirements_df = requirements_df.groupby("Requirement", as_index=False).mean()
    objections_df = objections_df.groupby("Objection", as_index=False).mean()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Score', y='Requirement', data=requirements_df, palette='Blues_d', hue='Requirement', legend=False)
    plt.title('Customer Requirements Scores')
    plt.xlabel('Score')
    plt.ylabel('Requirement')
    requirements_image_path = os.path.join('static', 'requirements.png')
    plt.savefig(requirements_image_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Score', y='Objection', data=objections_df, palette='Reds_d', hue='Objection', legend=False)
    plt.title('Customer Objections Scores')
    plt.xlabel('Score')
    plt.ylabel('Objection')
    objections_image_path = os.path.join('static', 'objections.png')
    plt.savefig(objections_image_path)
    plt.close()

    return requirements_image_path, objections_image_path


# Function to export data to CSV
def export_to_csv(data):
    requirements_df = pd.DataFrame([entry["Customer Requirements"] for entry in data])
    objections_df = pd.DataFrame([entry["Customer Objections"] for entry in data])

    requirements_csv = os.path.join('static', 'requirements.csv')
    objections_csv = os.path.join('static', 'objections.csv')

    requirements_df.to_csv(requirements_csv, index=False)
    objections_df.to_csv(objections_csv, index=False)

    return requirements_csv, objections_csv

# Function to export data to PDF
def export_to_pdf(data):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Customer Requirements and Objections Analysis", ln=True, align='C')

    for entry in data:
        pdf.multi_cell(0, 10, txt=json.dumps(entry, indent=4))

    pdf_path = os.path.join('static', 'analysis.pdf')
    pdf.output(pdf_path)

    return pdf_path

# Route for the upload form
@app.route('/')
def upload_form():
    return render_template('index.html')

# Route to handle file upload and processing
@app.route('/uploads', methods=['POST'])
def upload_file():
    files = request.files.getlist('files')
    if files:
        all_data = []
        for file in files:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            data = process_transcript(file_path)
            all_data.append(data)

        requirements_image_path, objections_image_path = vis(all_data)
        requirements_csv, objections_csv = export_to_csv(all_data)
        pdf_path = export_to_pdf(all_data)

        return render_template('result.html', 
                               requirements_image=requirements_image_path,
                               objections_image=objections_image_path,
                               data=json.dumps(all_data, indent=4),
                               requirements_csv=requirements_csv,
                               objections_csv=objections_csv,
                               pdf_path=pdf_path)
    return "No files uploaded", 400

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=2000)
