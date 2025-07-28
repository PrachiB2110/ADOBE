

# Persona-Driven PDF Analyzer

A CPU-only Python tool that analyzes PDF documents based on user personas and job-to-be-done contexts. The analyzer extracts relevant sections, scores their importance, and generates refined subsections for targeted document analysis.

## Features

- **Persona-Driven Analysis**: Scores document sections based on relevance to user persona and job requirements
- **Multi-Document Processing**: Handles multiple PDFs simultaneously
- **Intelligent Section Detection**: Automatically detects document sections using TOC or font analysis
- **CPU-Only Operation**: Designed to run without GPU dependencies
- **TF-IDF Similarity Scoring**: Uses advanced text similarity algorithms for relevance ranking
- **Dockerized Deployment**: Includes Docker configuration for consistent execution

## Prerequisites

- Docker with multi-platform support
- Python 3.9+ (if running locally)
- Input JSON configuration file
- PDF documents to analyze

## Installation & Setup

### Using Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build --platform=linux/amd64 -t challenge1b-cpu-only .
   ```

2. **Prepare your data structure:**
   ```
   your-project/
   ├── Collection_2/
   │   ├── challenge1b_input.json
   │   ├── PDFs/
   │   │   ├── document1.pdf
   │   │   ├── document2.pdf
   │   │   └── ...
   │   └── challenge1b_output.json (generated)
   ```

3. **Run the analyzer:**
   ```bash
   docker run --rm --platform=linux/amd64 \
     -v "$(pwd):/data" \
     challenge1b-cpu-only \
     python main.py \
     --input /data/Collection_2/challenge1b_input.json \
     --pdf_dir /data/Collection_2/PDFs \
     --output /data/Collection_2/challenge1b_output.json
   ```

### Local Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the analyzer:**
   ```bash
   python main.py \
     --input Collection_2/challenge1b_input.json \
     --pdf_dir Collection_2/PDFs \
     --output Collection_2/challenge1b_output.json
   ```

## Input Configuration

The input JSON file should follow this structure:

```json
{
  "challenge_info": {
    "description": "Challenge description",
    "requirements": "Specific requirements"
  },
  "persona": {
    "role": "Target user role",
    "background": "User background information",
    "goals": "User objectives",
    "pain_points": "User challenges"
  },
  "job_to_be_done": {
    "primary_goal": "Main objective",
    "context": "Usage context",
    "desired_outcome": "Expected results"
  },
  "documents": [
    {
      "filename": "document1.pdf",
      "description": "Document description"
    },
    {
      "filename": "document2.pdf", 
      "description": "Document description"
    }
  ]
}
```

## Output Format

The analyzer generates a JSON output with:

```json
{
  "metadata": {
    "input_documents": ["list of processed files"],
    "persona": "Combined persona text",
    "job_to_be_done": "Combined job context",
    "processing_timestamp": "ISO timestamp",
    "challenge_info": {}
  },
  "extracted_sections": [
    {
      "document": "filename.pdf",
      "section_title": "Section title",
      "importance_rank": 1,
      "page_number": 5
    }
  ],
  "subsection_analysis": [
    {
      "document": "filename.pdf",
      "refined_text": "Extracted relevant text...",
      "page_number": 5
    }
  ]
}
```

## Key Components

### PersonaDrivenAnalyzer Class

- **`extract_text_from_pdf()`**: Extracts structured text from PDF pages
- **`detect_sections()`**: Identifies document sections via TOC or font analysis
- **`score_section_relevance()`**: Calculates relevance scores using TF-IDF similarity
- **`extract_subsections()`**: Breaks sections into manageable chunks
- **`process_documents()`**: Main processing pipeline

### Scoring Algorithm

1. **TF-IDF Similarity**: Computes cosine similarity between section text and persona/job context
2. **Keyword Boosting**: Adds bonus scores for key terms found in sections
3. **Relevance Ranking**: Selects top 10 most relevant sections across all documents
4. **Subsection Refinement**: Extracts up to 5 refined text chunks for detailed analysis

## CPU-Only Architecture

The tool is specifically designed for CPU-only operation:

- Environment variables disable CUDA/GPU access
- Verification function ensures no GPU libraries are loaded
- Dependencies are installed with CPU-only flags
- Dockerfile purges any NVIDIA/CUDA packages

## Command Line Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--input` | Path to input JSON configuration file | Yes |
| `--pdf_dir` | Directory containing PDF documents | Yes |
| `--output` | Path for output JSON file | Yes |

## Dependencies

- **PyMuPDF (fitz)**: PDF text extraction
- **NLTK**: Natural language processing
- **scikit-learn**: TF-IDF vectorization and similarity
- **NumPy**: Numerical computations

## Error Handling

- Graceful handling of missing files or corrupted PDFs
- Fallback to simple keyword scoring if TF-IDF fails
- Error output generation maintains JSON structure
- Comprehensive logging of processing status

## Performance Considerations

- Text truncation (1000 chars) for similarity calculations
- Limited section extraction (top 10 sections, 5 subsections)
- Efficient memory usage with generators where possible
- CPU-optimized algorithms for large document processing

## Troubleshooting

### Common Issues

1. **Missing NLTK data**: The tool automatically downloads required NLTK datasets
2. **Docker platform issues**: Ensure `--platform=linux/amd64` is specified
3. **File permission errors**: Check that Docker has access to your data directory
4. **Memory issues**: For very large PDFs, consider splitting documents

### Debug Mode

Add debug logging by modifying the script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This tool is provided as-is for document analysis purposes. Ensure compliance with your organization's data handling policies when processing sensitive documents.
