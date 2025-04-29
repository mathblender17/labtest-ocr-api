from fastapi import FastAPI, File, UploadFile
from typing import List
import pytesseract
import re
import io
import numpy as np
from PIL import Image
import cv2
from pydantic import BaseModel
from spellchecker import SpellChecker

app = FastAPI()

# Define the response model
class LabTestData(BaseModel):
    test_name: str
    test_value: str
    bio_reference_range: str
    test_unit: str
    lab_test_out_of_range: bool

class ResponseModel(BaseModel):
    is_success: bool
    data: List[LabTestData]

# Function to preprocess the image for better OCR results
def preprocess_image(image: Image) -> Image:
    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Apply adaptive thresholding for better contrast
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return Image.fromarray(thresh)

# Function to process the image and extract text using Tesseract OCR
def ocr_image(image_file: UploadFile) -> str:
    image = Image.open(io.BytesIO(image_file.file.read()))
    preprocessed_image = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed_image)
    return text

# Function to clean the extracted OCR text
def clean_text(ocr_text: str) -> str:
    corrections = {
        'Blectanic': 'Electronics',
        'Catoulated': 'Calculated',
        'Hejan': 'Hospital',
        'Whale Blood': 'Whole Blood',
        # Add more common OCR errors here
    }
    for wrong, correct in corrections.items():
        ocr_text = ocr_text.replace(wrong, correct)
    return ocr_text

# Function to correct text using a spell checker
def correct_text(text: str) -> str:
    spell = SpellChecker()
    words = text.split()
    corrected_words = [spell.correction(word) for word in words]
    return ' '.join(corrected_words)

# Custom parser for lab test data after OCR extraction
def parse_lab_test_data(text: str) -> List[LabTestData]:
    # Define the regex pattern to match the test data format (value, unit, reference range)
    test_pattern = r'([A-Za-z\s\(\)]+)\s*[:\-]?\s*(\d+\.\d+|\d+)\s*([A-Za-z0-9/]+)\s*(\d+\.\d+-\d+\.\d+|\d+-\d+)'
    matches = re.findall(test_pattern, text)
    
    lab_tests = []
    
    # Process the matches and structure them
    for match in matches:
        test_name = match[0].strip()
        test_value = match[1].strip()
        test_unit = match[2].strip()
        reference_range = match[3].strip()
        
        # Check if the test value is within the reference range
        low, high = map(float, reference_range.split('-'))
        test_value_float = float(test_value)
        lab_test_out_of_range = test_value_float < low or test_value_float > high
        
        lab_tests.append(LabTestData(
            test_name=test_name,
            test_value=test_value,
            bio_reference_range=reference_range,
            test_unit=test_unit,
            lab_test_out_of_range=lab_test_out_of_range
        ))
    
    return lab_tests

# Endpoint to receive an image and return the lab test data
@app.post("/get-lab-tests", response_model=ResponseModel)
async def get_lab_tests(file: UploadFile = File(...)):
    try:
        # Step 1: Extract text from the image using OCR
        text = ocr_image(file)
        
        # Step 2: Clean and correct the OCR text
        cleaned_text = clean_text(text)
        
        # Step 3: Correct the text using spell checker
        corrected_text = correct_text(cleaned_text)
        
        # Step 4: Extract lab test data from the corrected OCR text using custom parser
        lab_tests = parse_lab_test_data(corrected_text)
        
        # Step 5: Return the structured JSON response
        return {"is_success": True, "data": lab_tests}
    
    except Exception as e:
        return {"is_success": False, "data": []}
