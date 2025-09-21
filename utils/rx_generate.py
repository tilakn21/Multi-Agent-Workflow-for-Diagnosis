from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import io
import os
from supabase import create_client, Client
from datetime import datetime
import uuid

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Prescription Image Generator API")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class TreatmentPlan(BaseModel):
    medications: List[str]
    lifestyle_modifications: List[str]

class Metadata(BaseModel):
    sex: Optional[str]
    body_system: Optional[str]
    symptom_tags: Optional[List[str]]
    abha_id: Optional[str]

class FinalDiagnosis(BaseModel):
    patient_problem: str
    diagnosis: List[str]
    diagnosis_simplified: Optional[List[str]] = []
    metadata: Metadata
    treatment_plan: TreatmentPlan

class GenerateImageRequest(BaseModel):
    final_diagnosis: FinalDiagnosis

class GenerateImageResponse(BaseModel):
    image_url: str

def text_wrap(text, font, max_width):
    lines = []
    words = text.split()
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    while words:
        line_words = []
        while words:
            line_words.append(words.pop(0))
            if words:
                test_line = ' '.join(line_words + words[:1])
                bbox = draw.textbbox((0,0), test_line, font=font)
                w = bbox[2] - bbox[0]
                if w > max_width:
                    break
        lines.append(' '.join(line_words))
    return lines

def create_prescription_header(draw, width, margin_left, margin_right):
    """Create prescription header similar to standard Rx format"""
    y = 30
    
    # Rx symbol - large font
    try:
        rx_font = ImageFont.truetype("arial.ttf", 48)
        header_font = ImageFont.truetype("arialbd.ttf", 16)
        text_font = ImageFont.truetype("arial.ttf", 12)
    except:
        rx_font = ImageFont.load_default()
        header_font = ImageFont.load_default() 
        text_font = ImageFont.load_default()
    
    # Draw Rx symbol
    draw.text((margin_left, y), "Rx", font=rx_font, fill=(0, 0, 0))
    
    # Patient ID (abha_id) at the top right
    info_x = width - 250
    draw.text((info_x, y), "Patient ID (ABHA): _________________________", font=text_font, fill=(0, 0, 0))

    # Draw line under header
    y = 120
    draw.line((margin_left, y, width - margin_right, y), fill=(0, 0, 0), width=2)

    return y + 20

def create_prescription_image(data: FinalDiagnosis) -> bytes:
    width, height = 800, 1000
    background_color = (255, 255, 255)  # White background like real prescription
    text_color = (0, 0, 0)  # Black text
    
    img = Image.new('RGB', (width, height), color=background_color)
    draw = ImageDraw.Draw(img)
    
    try:
        header_font = ImageFont.truetype("arialbd.ttf", 16)
        text_font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        header_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    margin_left, margin_right = 50, 50
    max_content_height = height - 200  # Reserve space for footer
    
    # Create header
    y = create_prescription_header(draw, width, margin_left, margin_right)
    
    # Patient ID and Date
    draw.text((margin_left, y), f"Patient ID: {data.metadata.abha_id or 'N/A'}", font=text_font, fill=text_color)
    draw.text((width - 200, y), f"Date: {datetime.now().strftime('%d-%b-%Y')}", font=text_font, fill=text_color)
    y += 35
    
    # Add separator line
    draw.line((margin_left, y, width - margin_right, y), fill=(0, 0, 0), width=1)
    y += 20
    
    # Check if we need multiple pages by estimating content height
    estimated_height = len(data.diagnosis) * 25 + len(data.treatment_plan.medications) * 25
    if data.diagnosis_simplified:
        estimated_height += len(data.diagnosis_simplified) * 25
    estimated_height += len(data.treatment_plan.lifestyle_modifications) * 25
    estimated_height += 200  # For patient summary
    
    if y + estimated_height > max_content_height:
        # Content will overflow, create multi-page
        return create_multi_page_prescription(data)
    
    # PRESCRIPTION: label (centered and bold)
    prescription_text = "PRESCRIPTION:"
    bbox = draw.textbbox((0, 0), prescription_text, font=header_font)
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) // 2, y), prescription_text, font=header_font, fill=text_color)
    y += 40
    
    # Patient Summary
    draw.text((margin_left, y), "Chief Complaint:", font=header_font, fill=text_color)
    y += 25
    summary_lines = text_wrap(data.patient_problem, text_font, width - margin_left - margin_right)
    for line in summary_lines[:3]:  # Limit to 3 lines
        draw.text((margin_left + 20, y), line, font=text_font, fill=text_color)
        y += 20
    y += 20
    
    # Diagnosis
    draw.text((margin_left, y), "Diagnosis:", font=header_font, fill=text_color)
    y += 25
    for i, diag in enumerate(data.diagnosis[:3]):  # Limit diagnoses
        draw.text((margin_left + 20, y), f"{i+1}. {diag}", font=text_font, fill=text_color)
        y += 22
    y += 20
    
    # Treatment/Medications
    draw.text((margin_left, y), "Treatment:", font=header_font, fill=text_color)
    y += 25
    for i, med in enumerate(data.treatment_plan.medications[:4]):  # Limit medications
        med_lines = text_wrap(med, text_font, width - margin_left - margin_right - 40)
        draw.text((margin_left + 20, y), f"{i+1}. {med_lines[0]}", font=text_font, fill=text_color)
        y += 22
        for additional_line in med_lines[1:2]:  # Only one additional line
            draw.text((margin_left + 40, y), additional_line, font=text_font, fill=text_color)
            y += 22
    
    # Footer section
    footer_y = height - 120
    draw.line((margin_left, footer_y, width - margin_right, footer_y), fill=(0, 0, 0), width=1)
    footer_y += 20
    
    # Instructions
    draw.text((margin_left, footer_y), "Follow-up:", font=small_font, fill=text_color)
    draw.text((margin_left + 100, footer_y), "as needed", font=small_font, fill=text_color)
    footer_y += 30
    
    # Signature line
    draw.text((margin_left, footer_y), "Doctor's Signature:", font=small_font, fill=text_color)
    draw.line((margin_left + 130, footer_y + 10, width - margin_right, footer_y + 10), fill=(0, 0, 0), width=1)
    
    bio = io.BytesIO()
    img.save(bio, format='PNG')
    bio.seek(0)
    return bio.read()

def create_multi_page_prescription(data: FinalDiagnosis) -> bytes:
    """Create a multi-page prescription when content overflows"""
    pages = []
    
    # Page 1 - Summary and primary info
    width, height = 800, 1000
    background_color = (255, 255, 255)
    text_color = (0, 0, 0)
    
    img1 = Image.new('RGB', (width, height), color=background_color)
    draw1 = ImageDraw.Draw(img1)
    
    try:
        header_font = ImageFont.truetype("arialbd.ttf", 16)
        text_font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        header_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    margin_left, margin_right = 50, 50
    
    # Page 1 content
    y = create_prescription_header(draw1, width, margin_left, margin_right)
    draw1.text((margin_left, y), f"Patient ID: {data.metadata.abha_id or 'N/A'}", font=text_font, fill=text_color)
    draw1.text((width - 200, y), f"Date: {datetime.now().strftime('%d-%b-%Y')}", font=text_font, fill=text_color)
    y += 35
    
    # Page indicator
    draw1.text((width - 100, height - 30), "Page 1 of 2", font=small_font, fill=text_color)
    
    # Save page 1
    bio1 = io.BytesIO()
    img1.save(bio1, format='PNG')
    
    # Page 2 - Detailed treatment
    img2 = Image.new('RGB', (width, height), color=background_color)
    draw2 = ImageDraw.Draw(img2)
    
    y = 50
    draw2.text((margin_left, y), "PRESCRIPTION (Continued)", font=header_font, fill=text_color)
    y += 40
    
    # Add all content to page 2
    draw2.text((margin_left, y), "Complete Treatment Plan:", font=header_font, fill=text_color)
    y += 30
    
    for med in data.treatment_plan.medications:
        med_lines = text_wrap(med, text_font, width - margin_left - margin_right - 40)
        for line in med_lines:
            draw2.text((margin_left + 20, y), f"• {line}", font=text_font, fill=text_color)
            y += 22
    
    draw2.text((width - 100, height - 30), "Page 2 of 2", font=small_font, fill=text_color)
    
    # For simplicity, return first page only
    # In a real implementation, you would combine both images or handle multi-page differently
    bio1.seek(0)
    return bio1.read()

# @app.post("/generate_report_image", response_model=GenerateImageResponse)
# async def generate_report_image(req: GenerateImageRequest):
#     print("[DEBUG] starting image generation")
#     try:
#         img_bytes = create_prescription_image(req.final_diagnosis)
#         print("[DEBUG] image created successfully")
#     except Exception as e:
#         print(f"[ERROR] Image generation failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

#     file_name = f"prescription_{uuid.uuid4().hex}.png"
#     try:
#         res = supabase.storage.from_('GDHS').upload(file_name, img_bytes)
#         print("Supabase upload response:", res)
#         error = getattr(res, "error", None)
#         if error:
#             message = getattr(error, "message", str(error))
#             raise Exception(message)
#         url = f"{SUPABASE_URL}/storage/v1/object/public/GDHS/{file_name}"
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Uploading to Supabase failed: {e}")

#     if not url or not isinstance(url, str):
#         raise HTTPException(status_code=500, detail="Image upload did not return a valid URL")

#     return GenerateImageResponse(image_url=url)
