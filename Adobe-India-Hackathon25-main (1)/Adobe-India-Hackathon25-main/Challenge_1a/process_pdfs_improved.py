import fitz
import os
import json
import re
from collections import defaultdict

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

def extract_title_from_pdf(pdf_path):
    """
    Extract the document title from PDF metadata or first page content.
    """
    doc = fitz.open(pdf_path)
    
    # Try to get title from metadata first
    metadata = doc.metadata
    if metadata.get("title") and metadata["title"].strip():
        return metadata["title"].strip()
    
    # Fallback: extract from first page content
    if len(doc) > 0:
        page = doc[0]
        # Get text blocks and find the largest/first text that looks like a title
        text_dict = page.get_text("dict")
        
        title_candidates = []
        
        for block in text_dict["blocks"]:
            if block["type"] != 0:  # Skip image blocks
                continue
                
            for line in block["lines"]:
                line_text = ""
                max_font_size = 0
                
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        line_text += text + " "
                        max_font_size = max(max_font_size, span["size"])
                
                line_text = line_text.strip()
                if line_text and len(line_text) > 3:  # Avoid single characters
                    title_candidates.append({
                        "text": line_text,
                        "font_size": max_font_size,
                        "y_pos": line["bbox"][1]  # y-position for ordering
                    })
        
        # Sort by y-position (top to bottom) and font size
        title_candidates.sort(key=lambda x: (x["y_pos"], -x["font_size"]))
        
        # Return the first substantial text (likely the title)
        for candidate in title_candidates:
            if len(candidate["text"]) >= 10:  # Reasonable title length
                return candidate["text"]
        
        # If no good candidate, return the first text found
        if title_candidates:
            return title_candidates[0]["text"]
    
    doc.close()
    return "Untitled Document"

def detect_heading_levels(spans_data):
    """
    Analyze font sizes and formatting to determine heading levels.
    """
    # Collect all font sizes
    font_sizes = []
    for span_info in spans_data:
        if span_info["font_size"]:
            font_sizes.append(span_info["font_size"])
    
    if not font_sizes:
        return {}
    
    # Get unique font sizes sorted in descending order
    unique_sizes = sorted(set(font_sizes), reverse=True)
    
    # Map font sizes to heading levels
    level_mapping = {}
    for i, size in enumerate(unique_sizes[:6]):  # Max 6 heading levels
        if i == 0:
            level_mapping[size] = "H1"
        elif i == 1:
            level_mapping[size] = "H2" 
        elif i == 2:
            level_mapping[size] = "H3"
        elif i == 3:
            level_mapping[size] = "H4"
        elif i == 4:
            level_mapping[size] = "H5"
        else:
            level_mapping[size] = "H6"
    
    return level_mapping

def is_likely_heading(text, font_size, is_bold, level_mapping):
    """
    Determine if a text span is likely a heading based on various criteria.
    """
    # Check if font size corresponds to a heading level
    if font_size in level_mapping:
        # Additional criteria for headings
        text = text.strip()
        
        # Length criteria (headings are usually shorter)
        if len(text) > 200:
            return False
            
        # Pattern matching for common heading patterns
        heading_patterns = [
            r'^\d+\.?\s+',  # Numbered headings (1. , 2.1 )
            r'^[A-Z][a-z]+',  # Capitalized words
            r'^[A-Z\s]+$',  # All caps
            r'^(Chapter|Section|Part|Appendix)',  # Common heading words
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True
                
        # If bold and reasonable length, likely a heading
        if is_bold and 5 <= len(text) <= 100:
            return True
            
        # If font size is significantly larger than body text
        return True
    
    return False

def extract_outline_from_pdf(pdf_path):
    """
    Extract document outline/headings from PDF.
    """
    doc = fitz.open(pdf_path)
    
    # First, try to get outline from PDF bookmarks/TOC
    toc = doc.get_toc()
    if toc:
        outline = []
        for item in toc:
            level = item[0]  # Outline level
            title = item[1]  # Title text
            page_num = item[2]  # Page number
            
            # Convert level to H1, H2, etc.
            heading_level = f"H{min(level, 6)}"
            
            outline.append({
                "level": heading_level,
                "text": title.strip(),
                "page": page_num
            })
        doc.close()
        return outline
    
    # Fallback: extract from text analysis
    spans_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get text with formatting information
        text_dict = page.get_text("dict")
        
        for block in text_dict["blocks"]:
            if block["type"] != 0:  # Skip image blocks
                continue
                
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                        
                    spans_data.append({
                        "text": text,
                        "font_size": span["size"],
                        "font_name": span["font"],
                        "bold": "Bold" in span["font"] or "bold" in span["font"].lower(),
                        "page": page_num + 1,
                        "y_pos": span["bbox"][1]
                    })
    
    doc.close()
    
    # Detect heading levels based on font sizes
    level_mapping = detect_heading_levels(spans_data)
    
    # Extract headings
    outline = []
    seen_headings = set()  # To avoid duplicates
    
    for span_info in spans_data:
        if is_likely_heading(
            span_info["text"], 
            span_info["font_size"], 
            span_info["bold"], 
            level_mapping
        ):
            text = span_info["text"].strip()
            
            # Avoid duplicate headings
            heading_key = (text.lower(), span_info["page"])
            if heading_key in seen_headings:
                continue
            seen_headings.add(heading_key)
            
            # Clean up heading text
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            
            if len(text) >= 3:  # Minimum heading length
                outline.append({
                    "level": level_mapping[span_info["font_size"]],
                    "text": text,
                    "page": span_info["page"]
                })
    
    # Sort outline by page number and y-position
    outline.sort(key=lambda x: x["page"])
    
    return outline

def process_single_pdf(pdf_path, output_path):
    """
    Process a single PDF and generate JSON output according to schema.
    """
    try:
        # Extract title
        title = extract_title_from_pdf(pdf_path)
        
        # Extract outline
        outline = extract_outline_from_pdf(pdf_path)
        
        # Create output according to schema
        output_data = {
            "title": title,
            "outline": outline
        }
        
        # Write JSON output
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        return True
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        
        # Create minimal valid output on error
        error_output = {
            "title": "Processing Error",
            "outline": []
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(error_output, f, indent=2, ensure_ascii=False)
            
        return False

def process_all_pdfs(input_dir, output_dir):
    """
    Process all PDFs in the input directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
        
        print(f"Processing: {filename}")
        
        success = process_single_pdf(pdf_path, output_path)
        
        if success:
            processed_count += 1
        else:
            error_count += 1
    
    print(f"Processing complete. Success: {processed_count}, Errors: {error_count}")

if __name__ == "__main__":
    process_all_pdfs(INPUT_DIR, OUTPUT_DIR)
    print("Done.")
