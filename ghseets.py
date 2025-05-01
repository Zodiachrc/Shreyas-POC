import os
import requests
from together import Together
from PyPDF2 import PdfReader
import re

# Initialize Together client
client = Together()

# Google Sheets API URL
url = "https://script.google.com/macros/s/AKfycbxhlrzKsmqQhQZMfcgffXsa82idww7442o8_3TEyv5vvbqPo5BVEvgNvNLmFWXsNbnl/exec"

# Folder containing resumes
resume_folder = r"C:\Users\andsh\Downloads\zodiac resumes"

# Queries for extracting fields
queries = {
    "Name": "What is the candidate's name?",
    "Qualification": "What is this guy's latest qualification?",
    "Skills": "What are his technical skills?",
    "Experience": "What is the experience in years?",
    "Companies": "What are the companies he worked at?",
    "Location": "What is the location of this man?",
    "Certificates": "List any certificates, courses, bootcamps, or training programs this person has completed.",
    "Notice Period": "What is his notice period?"
}

# Loop through each PDF in the folder
for filename in os.listdir(resume_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(resume_folder, filename)
        
        # Extract text from the PDF
        reader = PdfReader(pdf_path)
        context = " ".join(page.extract_text() for page in reader.pages if page.extract_text())

        # Prepare a single prompt for all queries
        questions = "\n".join([f"{field}: {query}" for field, query in queries.items()])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant helping with resume data extraction. "
                    "Answer briefly, directly, and without any explanation or commentary. "
                    "If the answer is e.g., a number or a date or a bunch of certificates, return only that. "
                    "Do NOT restate the question. Do NOT explain the answer. Return just the answer. "
                    "The current year is May 2025."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestions:\n{questions}\nAnswers:"
            }
        ]

        # Get the response from the AI model
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=messages,
            max_tokens=1024,
            temperature=0.6
        )

        # Extract the actual answers from the response
        response_text = response.choices[0].message.content.strip()

        # Find the position of the </think> tag
        end_tag = "</think>"
        end_index = response_text.find(end_tag)

        if end_index != -1:
            # Extract everything after the </think> tag
            final_answer = response_text[end_index + len(end_tag):].strip()
        else:
            # If </think> is not found, use the entire response
            final_answer = response_text.strip()

        # Parse the answers into a dictionary
        extracted_data = {}
        for line in final_answer.split("\n"):
            if ":" in line:
                field, answer = line.split(":", 1)
                extracted_data[field.strip()] = answer.strip()

        # Skip if no valid data
        if not extracted_data:
            print(f"No valid data extracted for {filename}. Skipping...")
            continue

        # Send cleaned data to Google Sheets
        response = requests.post(url, data=extracted_data)

        print("Data Sent to Google Sheets:", extracted_data)
        print(f"Processed {filename}")
        print("Status Code:", response.status_code)
        print("Response:", response.text)



