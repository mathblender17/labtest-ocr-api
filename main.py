from fastapi import FastAPI, File, UploadFile
from typing import List
import pytesseract
import re
import io
import numpy as np
from PIL import Image
from pydantic import BaseModel

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

# Function to process the image and extract text using Tesseract OCR
def ocr_image(image_file: UploadFile) -> str:
    image = Image.open(io.BytesIO(image_file.file.read()))
    text = pytesseract.image_to_string(image)
    return text

# Function to clean and parse the extracted OCR text
def extract_lab_test_data(text: str) -> List[LabTestData]:
    # Define regular expressions to capture test names, values, units, and reference ranges
    test_pattern = r'([A-Za-z\s\(\)]+)\s*[:\-]?\s*(\d+\.\d+|\d+)\s*([A-Za-z/]+)\s*(\d+\.\d+-\d+\.\d+)'
    #test_pattern = r'([A-Za-z0-9\s\(\)\-/]+?)[:\-]?\s*(\d+(?:\.\d+)?)\s*([a-zA-Z%/]+)?\s*(\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?)'

    matches = re.findall(test_pattern, text)
    
    lab_tests = []
    
    # Parse the matches and create LabTestData objects
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
        
        # Step 2: Extract lab test data from the OCR text
        lab_tests = extract_lab_test_data(text)
        
        # Step 3: Return the structured JSON response
        return {"is_success": True, "data": lab_tests}
    
    except Exception as e:
        return {"is_success": False, "data": []}
