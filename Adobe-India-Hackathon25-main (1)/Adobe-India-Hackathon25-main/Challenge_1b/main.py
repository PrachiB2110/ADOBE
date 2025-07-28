import fitz
import os
import json
import re
import argparse
from collections import defaultdict
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NVIDIA_VISIBLE_DEVICES'] = ''

def verify_cpu_only():
    forbidden_modules = ['cupy', 'torch', 'tensorflow-gpu', 'nvidia']
    loaded_gpu_modules = []
    import sys
    for module_name in sys.modules:
        for forbidden in forbidden_modules:
            if forbidden in module_name.lower():
                loaded_gpu_modules.append(module_name)
    if loaded_gpu_modules:
        print(f"WARNING: GPU modules detected: {loaded_gpu_modules}")
    else:
        print("✓ CPU-only mode verified - no GPU libraries loaded")

verify_cpu_only()

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class PersonaDrivenAnalyzer:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        sections = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            page_text = ""

            for block in text_dict["blocks"]:
                if block["type"] == 0:
                    block_text = ""
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        block_text += line_text + " "
                    page_text += block_text + "\n"

            if page_text.strip():
                sections.append({
                    "page_number": page_num + 1,
                    "text": page_text.strip()
                })

        doc.close()
        return sections

    def detect_sections(self, pdf_path):
        doc = fitz.open(pdf_path)
        sections = []

        toc = doc.get_toc()
        if toc:
            for item in toc:
                level, title, page_num = item
                sections.append({
                    "title": title.strip(),
                    "page_number": page_num,
                    "level": level,
                    "type": "toc_section"
                })

        if not sections:
            font_sizes = []
            text_blocks = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text_dict = page.get_text("dict")

                for block in text_dict["blocks"]:
                    if block["type"] == 0:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    font_sizes.append(span["size"])
                                    text_blocks.append({
                                        "text": text,
                                        "font_size": span["size"],
                                        "font": span["font"],
                                        "page": page_num + 1,
                                        "bold": "Bold" in span["font"] or "bold" in span["font"].lower()
                                    })

            if font_sizes:
                unique_sizes = sorted(set(font_sizes), reverse=True)
                heading_sizes = unique_sizes[:3]

                for block in text_blocks:
                    if (block["font_size"] in heading_sizes and 
                        5 < len(block["text"]) < 200 and
                        not block["text"].endswith('.')):

                        sections.append({
                            "title": block["text"],
                            "page_number": block["page"],
                            "level": heading_sizes.index(block["font_size"]) + 1,
                            "type": "detected_section"
                        })

        doc.close()
        return sections

    def score_section_relevance(self, section_text, persona_text, job_text):
        context_text = f"{persona_text} {job_text}"

        try:
            documents = [section_text[:1000], context_text[:1000]]  # truncate long text
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            key_terms = self.extract_key_terms(persona_text, job_text)
            section_lower = section_text.lower()
            boost_score = sum(0.1 for term in key_terms if term.lower() in section_lower)

            return min(similarity + boost_score, 1.0)

        except Exception:
            return self.simple_keyword_score(section_text, context_text)

    def extract_key_terms(self, persona_text, job_text):
        combined_text = f"{persona_text} {job_text}"
        words = word_tokenize(combined_text.lower())
        return list({word for word in words if len(word) > 3 and word not in self.stop_words and word.isalpha()})

    def simple_keyword_score(self, section_text, context_text):
        section_words = {w for w in word_tokenize(section_text.lower()) if w not in self.stop_words and len(w) > 3}
        context_words = {w for w in word_tokenize(context_text.lower()) if w not in self.stop_words and len(w) > 3}
        if not context_words:
            return 0.0
        return len(section_words & context_words) / len(context_words)

    def extract_subsections(self, section_text, max_length=500):
        sentences = sent_tokenize(section_text)
        subsections = []
        current_subsection = ""

        for sentence in sentences:
            if len(current_subsection + sentence) <= max_length:
                current_subsection += sentence + " "
            else:
                if current_subsection.strip():
                    subsections.append(current_subsection.strip())
                current_subsection = sentence + " "

        if current_subsection.strip():
            subsections.append(current_subsection.strip())

        return subsections

    def process_documents(self, input_file_path):
        with open(input_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        challenge_info = config.get("challenge_info", {})
        documents = config.get("documents", [])
        persona = config.get("persona", {})
        job_to_be_done = config.get("job_to_be_done", {})

        persona_text = " ".join(str(v) for v in persona.values() if v)
        job_text = " ".join(str(v) for v in job_to_be_done.values() if v)

        all_section_scores = []
        for doc_info in documents:
            filename = doc_info.get("filename", "")
            pdf_path = os.path.join(self.input_dir, filename)
            if not os.path.exists(pdf_path):
                continue

            detected_sections = self.detect_sections(pdf_path)
            text_sections = self.extract_text_from_pdf(pdf_path)

            if not detected_sections:
                for i, page_content in enumerate(text_sections):
                    detected_sections.append({
                        "title": f"Page {page_content['page_number']} Content",
                        "page_number": page_content['page_number'],
                        "level": 1,
                        "type": "page_section"
                    })

            for section in detected_sections:
                section_page = section["page_number"]
                section_text = next((t["text"] for t in text_sections if t["page_number"] == section_page), "")
                if section_text:
                    score = self.score_section_relevance(section_text, persona_text, job_text)
                    all_section_scores.append({
                        "document": filename,
                        "section_title": section["title"],
                        "page_number": section_page,
                        "score": score,
                        "text": section_text
                    })

        # Select top 5 sections across all documents
        top_sections = sorted(all_section_scores, key=lambda x: x["score"], reverse=True)

        all_sections = []
        all_subsections = []
        for rank, section in enumerate(top_sections, 1):
            if rank > 10:
                break  # Enforce importance_rank <= 10

            all_sections.append({
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": rank,
                "page_number": section["page_number"]
            })

            if len(all_subsections) < 5:
                subsections = self.extract_subsections(section["text"])
                for subsec_text in subsections[:3]:
                    if len(all_subsections) < 5:
                        all_subsections.append({
                            "document": section["document"],
                            "refined_text": subsec_text,
                            "page_number": section["page_number"]
                        })
                    else:
                        break

        return {
            "metadata": {
                "input_documents": [doc.get("filename", "") for doc in documents],
                "persona": persona_text,
                "job_to_be_done": job_text,
                "processing_timestamp": datetime.now().isoformat(),
                "challenge_info": challenge_info
            },
            "extracted_sections": all_sections,
            "subsection_analysis": all_subsections
        }


def main():
    parser = argparse.ArgumentParser(description="PDF Analyzer")
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--pdf_dir', required=True, help='PDFs directory')
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()

    analyzer = PersonaDrivenAnalyzer(args.pdf_dir)

    try:
        result = analyzer.process_documents(args.input)

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"✓ Processing complete. Output written to {args.output}")
        print(f"✓ Processed {len(result['metadata']['input_documents'])} documents")
        print(f"✓ Extracted {len(result['extracted_sections'])} sections")
        print(f"✓ Generated {len(result['subsection_analysis'])} refined subsections")

    except Exception as e:
        print(f"✗ Error during processing: {str(e)}")

        error_output = {
            "metadata": {
                "input_documents": [],
                "persona": "Error occurred",
                "job_to_be_done": "Processing failed",
                "processing_timestamp": datetime.now().isoformat(),
                "error": str(e)
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(error_output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
    print("Done.")