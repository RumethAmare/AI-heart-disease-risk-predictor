#!/usr/bin/env python3
"""
PDF Reader Script for extracting text from the project instructions document.
"""

import PyPDF2
import os

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"PDF has {len(pdf_reader.pages)} pages")
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
                print(f"Extracted text from page {page_num + 1}")
                
        return text
        
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None

def save_text_to_file(text, output_path):
    """
    Save extracted text to a text file.
    
    Args:
        text (str): Text content to save
        output_path (str): Output file path
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text saved to: {output_path}")
    except Exception as e:
        print(f"Error saving text: {str(e)}")

if __name__ == "__main__":
    # Path to the PDF file
    pdf_file = r"c:\Users\Dineth\Desktop\FDM - reading\FDM\FDM project\FDM - IT3051- Mini Project - 2025 - finalized instructions.pdf"
    
    # Output text file
    output_file = r"c:\Users\Dineth\Desktop\FDM - reading\FDM\FDM project\project_instructions.txt"
    
    if os.path.exists(pdf_file):
        print(f"Reading PDF: {pdf_file}")
        extracted_text = extract_text_from_pdf(pdf_file)
        
        if extracted_text:
            save_text_to_file(extracted_text, output_file)
            print("\nFirst 1000 characters of extracted text:")
            print("-" * 50)
            print(extracted_text[:1000])
            print("-" * 50)
        else:
            print("Failed to extract text from PDF")
    else:
        print(f"PDF file not found: {pdf_file}")