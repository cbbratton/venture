"""
Executive Summary Analyzer
A comprehensive web application for analyzing company executive summaries using LLM
"""

import os
import logging
import tempfile # noqa
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path #noqa

# Web framework
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS

# PDF processing
import PyPDF2
from pdfplumber import PDF

# LLM integration
import openai #noqa
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Additional utilities
import markdown # noqa
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Enable CORS for API endpoints
CORS(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Data models
@dataclass
class ExtractedInformation:
    """Data model for extracted company information"""
    company_name: Optional[str] = None
    technology_type: Optional[str] = None
    need_addressed: Optional[str] = None
    market_size: Optional[str] = None
    market_calculation_method: Optional[str] = None
    product_development_stage: Optional[str] = None
    current_sales: Optional[str] = None
    exit_value_range: Optional[str] = None
    years_to_exit: Optional[str] = None
    investment_needed: Optional[str] = None
    missing_skills: Optional[str] = None
    extraction_timestamp: str = ""
    confidence_scores: Dict[str, float] = None

    def __post_init__(self):
        if not self.extraction_timestamp:
            self.extraction_timestamp = datetime.now().isoformat()
        if self.confidence_scores is None:
            self.confidence_scores = {}

@dataclass
class Report:
    """Data model for the final report"""
    nature_and_state: str
    market_need_and_size: str
    roi_elements: str
    management_team_strength: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0'
            }

# LLM Configuration
class LLMAnalyzer:
    """Handles LLM operations for document analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            model_name="gpt-4",
            temperature=0.1,
            max_tokens=2000
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
    
    def extract_information(self, text: str) -> ExtractedInformation:
        """Extract structured information from document text"""
        
        # Split text if too long
        chunks = self.text_splitter.split_text(text)
        
        # Prepare the extraction prompt
        extraction_prompt = """
        You are an expert business analyst. Extract the following information from the executive summary.
        If information is not available, clearly state "Information not provided in the document."
        
        Questions to answer:
        1. What is the company name?
        2. Technology type (Device, diagnostic, therapeutic, or digital health)
        3. What is the need addressed by the product?
        4. How large is the potential market?
        5. Was that potential calculated "top down" or "bottom up"?
        6. How developed is the product? (concept only, prototype, in testing, or available on the market)
        7. If it is on the market, what is the current level of sales?
        8. What is the range of the potential value that the company might realize upon exit?
        9. How many years to that exit?
        10. How much money must be invested to secure that exit?
        11. What skills needed to execute the plan are missing from the management team?
        
        Provide your response in JSON format with the following keys:
        - company_name
        - technology_type
        - need_addressed
        - market_size
        - market_calculation_method
        - product_development_stage
        - current_sales
        - exit_value_range
        - years_to_exit
        - investment_needed
        - missing_skills
        
        Also include a confidence_score (0-1) for each field indicating how confident you are in the extraction.
        """
        
        # Process chunks and combine results
        all_extractions = []
        
        for chunk in chunks[:3]:  # Process up to 3 chunks to manage costs
            messages = [
                SystemMessage(content=extraction_prompt),
                HumanMessage(content=f"Executive Summary:\n{chunk}")
            ]
            
            try:
                response = self.llm.invoke(messages)
                result = json.loads(response.content)
                all_extractions.append(result)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                continue
        
        # Merge results from multiple chunks
        merged_info = self._merge_extractions(all_extractions)
        
        return ExtractedInformation(
            company_name=merged_info.get('company_name'),
            technology_type=merged_info.get('technology_type'),
            need_addressed=merged_info.get('need_addressed'),
            market_size=merged_info.get('market_size'),
            market_calculation_method=merged_info.get('market_calculation_method'),
            product_development_stage=merged_info.get('product_development_stage'),
            current_sales=merged_info.get('current_sales'),
            exit_value_range=merged_info.get('exit_value_range'),
            years_to_exit=merged_info.get('years_to_exit'),
            investment_needed=merged_info.get('investment_needed'),
            missing_skills=merged_info.get('missing_skills'),
            confidence_scores=merged_info.get('confidence_scores', {})
        )
    
    def _merge_extractions(self, extractions: List[Dict]) -> Dict:
        """Merge multiple extraction results, preferring non-null values with higher confidence"""
        if not extractions:
            return {}
        
        if len(extractions) == 1:
            return extractions[0]
        
        merged = {}
        confidence_scores = {}
        
        fields = [
            'company_name', 'technology_type', 'need_addressed', 'market_size',
            'market_calculation_method', 'product_development_stage', 'current_sales',
            'exit_value_range', 'years_to_exit', 'investment_needed', 'missing_skills'
        ]
        
        for field in fields:
            best_value = None
            best_confidence = 0
            
            for extraction in extractions:
                value = extraction.get(field)
                confidence = extraction.get('confidence_scores', {}).get(field, 0.5)
                
                if value and "not provided" not in value.lower() and confidence > best_confidence:
                    best_value = value
                    best_confidence = confidence
            
            merged[field] = best_value or "Information not provided in the document."
            confidence_scores[field] = best_confidence
        
        merged['confidence_scores'] = confidence_scores
        return merged
    
    def generate_report(self, info: ExtractedInformation) -> Report:
        """Generate a structured report from extracted information"""
        
        report_prompt = """
        Based on the extracted information, create a professional investment analysis report.
        Use the following headers and create comprehensive paragraphs for each section:
        
        1. Nature and state of the product
        2. Market Need and Size
        3. Elements of potential ROI
        4. Strength of the Management Team
        
        Make the report professional, analytical, and suitable for investors.
        If information is missing, acknowledge it professionally.
        """
        
        info_text = f"""
        Company: {info.company_name}
        Technology Type: {info.technology_type}
        Need Addressed: {info.need_addressed}
        Market Size: {info.market_size}
        Market Calculation: {info.market_calculation_method}
        Development Stage: {info.product_development_stage}
        Current Sales: {info.current_sales}
        Exit Value: {info.exit_value_range}
        Years to Exit: {info.years_to_exit}
        Investment Needed: {info.investment_needed}
        Missing Skills: {info.missing_skills}
        """
        
        messages = [
            SystemMessage(content=report_prompt),
            HumanMessage(content=info_text)
        ]
        
        try:
            response = self.llm.invoke(messages)
            sections = self._parse_report_sections(response.content)
            
            return Report(
                nature_and_state=sections.get('nature_and_state', ''),
                market_need_and_size=sections.get('market_need_and_size', ''),
                roi_elements=sections.get('roi_elements', ''),
                management_team_strength=sections.get('management_team_strength', ''),
                metadata={
                    'company_name': info.company_name,
                    'extraction_timestamp': info.extraction_timestamp,
                    'confidence_scores': info.confidence_scores
                }
            )
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def _parse_report_sections(self, report_text: str) -> Dict[str, str]:
        """Parse report text into sections"""
        sections = {
            'nature_and_state': '',
            'market_need_and_size': '',
            'roi_elements': '',
            'management_team_strength': ''
        }
        
        # Simple parsing based on section headers
        current_section = None
        lines = report_text.split('\n')
        
        section_mapping = {
            'nature and state': 'nature_and_state',
            'market need': 'market_need_and_size',
            'roi': 'roi_elements',
            'management': 'management_team_strength'
        }
        
        for line in lines:
            lower_line = line.lower()
            
            # Check if this line is a section header
            for key, section_key in section_mapping.items():
                if key in lower_line:
                    current_section = section_key
                    break
            
            # Add content to current section
            if current_section and line.strip() and not any(key in lower_line for key in section_mapping):
                sections[current_section] += line + '\n'
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections

# PDF Processing
class PDFProcessor:
    """Handles PDF file processing and text extraction"""
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """Extract text from PDF file using multiple methods for robustness"""
        text = ""
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with PDF.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Fallback to PyPDF2 if needed
        if not text.strip():
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n"
            except Exception as e:
                logger.error(f"PyPDF2 extraction failed: {e}")
                raise ValueError("Unable to extract text from PDF")
        
        return text.strip()

# Report Generator
class ReportGenerator:
    """Handles report generation in various formats"""
    
    @staticmethod
    def generate_html(report: Report, info: ExtractedInformation) -> str:
        """Generate HTML report"""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Company Analysis Report - {company_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f4f4f4;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #007bff;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #007bff;
                    margin-top: 30px;
                }}
                .metadata {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 30px;
                }}
                .section {{
                    margin-bottom: 25px;
                }}
                .info-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                    margin-bottom: 30px;
                }}
                .info-item {{
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .info-label {{
                    font-weight: bold;
                    color: #666;
                }}
                .confidence {{
                    font-size: 0.8em;
                    color: #888;
                }}
                .timestamp {{
                    text-align: right;
                    color: #666;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Investment Analysis Report</h1>
                
                <div class="metadata">
                    <h3>{company_name}</h3>
                    <div class="timestamp">Generated: {timestamp}</div>
                </div>
                
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Technology Type</div>
                        <div>{technology_type}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Development Stage</div>
                        <div>{development_stage}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Market Size</div>
                        <div>{market_size}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Investment Needed</div>
                        <div>{investment_needed}</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Nature and State of the Product</h2>
                    <p>{nature_and_state}</p>
                </div>
                
                <div class="section">
                    <h2>Market Need and Size</h2>
                    <p>{market_need_and_size}</p>
                </div>
                
                <div class="section">
                    <h2>Elements of Potential ROI</h2>
                    <p>{roi_elements}</p>
                </div>
                
                <div class="section">
                    <h2>Strength of the Management Team</h2>
                    <p>{management_team_strength}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return template.format(
            company_name=info.company_name or "Unknown Company",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            technology_type=info.technology_type or "Not specified",
            development_stage=info.product_development_stage or "Not specified",
            market_size=info.market_size or "Not specified",
            investment_needed=info.investment_needed or "Not specified",
            nature_and_state=report.nature_and_state.replace('\n', '<br>'),
            market_need_and_size=report.market_need_and_size.replace('\n', '<br>'),
            roi_elements=report.roi_elements.replace('\n', '<br>'),
            management_team_strength=report.management_team_strength.replace('\n', '<br>')
        )
    
    @staticmethod
    def generate_markdown(report: Report, info: ExtractedInformation) -> str:
        """Generate Markdown report"""
        template = """# Investment Analysis Report

## Company: {company_name}

**Generated:** {timestamp}

### Quick Facts
- **Technology Type:** {technology_type}
- **Development Stage:** {development_stage}
- **Market Size:** {market_size}
- **Investment Needed:** {investment_needed}
- **Exit Timeline:** {exit_timeline}
- **Exit Value Range:** {exit_value}

---

## Nature and State of the Product

{nature_and_state}

## Market Need and Size

{market_need_and_size}

## Elements of Potential ROI

{roi_elements}

## Strength of the Management Team

{management_team_strength}

---

### Extracted Information Summary

| Field | Value | Confidence |
|-------|-------|------------|
| Company Name | {company_name} | {conf_company} |
| Technology Type | {technology_type} | {conf_tech} |
| Need Addressed | {need_addressed} | {conf_need} |
| Market Calculation | {market_calc} | {conf_market_calc} |
| Current Sales | {current_sales} | {conf_sales} |
| Missing Skills | {missing_skills} | {conf_skills} |
"""
        
        return template.format(
            company_name=info.company_name or "Unknown Company",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            technology_type=info.technology_type or "Not specified",
            development_stage=info.product_development_stage or "Not specified",
            market_size=info.market_size or "Not specified",
            investment_needed=info.investment_needed or "Not specified",
            exit_timeline=info.years_to_exit or "Not specified",
            exit_value=info.exit_value_range or "Not specified",
            nature_and_state=report.nature_and_state,
            market_need_and_size=report.market_need_and_size,
            roi_elements=report.roi_elements,
            management_team_strength=report.management_team_strength,
            need_addressed=info.need_addressed or "Not specified",
            market_calc=info.market_calculation_method or "Not specified",
            current_sales=info.current_sales or "Not specified",
            missing_skills=info.missing_skills or "Not specified",
            conf_company=f"{info.confidence_scores.get('company_name', 0):.2f}",
            conf_tech=f"{info.confidence_scores.get('technology_type', 0):.2f}",
            conf_need=f"{info.confidence_scores.get('need_addressed', 0):.2f}",
            conf_market_calc=f"{info.confidence_scores.get('market_calculation_method', 0):.2f}",
            conf_sales=f"{info.confidence_scores.get('current_sales', 0):.2f}",
            conf_skills=f"{info.confidence_scores.get('missing_skills', 0):.2f}"
        )

# Flask Routes
@app.route('/')
def index():
    """Serve the main upload page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    """API endpoint for document analysis"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Extract text from PDF
        logger.info(f"Processing file: {unique_filename}")
        pdf_processor = PDFProcessor()
        text = pdf_processor.extract_text(file_path)
        
        if not text:
            return jsonify({'error': 'Unable to extract text from PDF'}), 400
        
        # Analyze with LLM
        analyzer = LLMAnalyzer()
        extracted_info = analyzer.extract_information(text)
        report = analyzer.generate_report(extracted_info)
        
        # Generate reports in different formats
        report_generator = ReportGenerator()
        html_report = report_generator.generate_html(report, extracted_info)
        markdown_report = report_generator.generate_markdown(report, extracted_info)
        
        # Save reports
        report_base = f"reports/{timestamp}_{extracted_info.company_name or 'unknown'}"
        html_path = f"{report_base}.html"
        md_path = f"{report_base}.md"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        # Return results
        return jsonify({
            'success': True,
            'extracted_info': asdict(extracted_info),
            'report': asdict(report),
            'report_files': {
                'html': html_path,
                'markdown': md_path
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/report/<format>/<path:filename>')
def download_report(format, filename):
    """Download generated report"""
    try:
        file_path = f"reports/{filename}"
        if not os.path.exists(file_path):
            return jsonify({'error': 'Report not found'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=os.path.basename(file_path)
        )
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# Utility functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Create templates directory and index.html
def create_templates():
    """Create HTML templates"""
    os.makedirs('templates', exist_ok=True)
    
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Summary Analyzer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            background: white;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            width: 90%;
            max-width: 600px;
        }
        
        h1 {
            color: #333;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 0.5rem;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background-color: #f8f9ff;
        }
        
        .upload-area.dragging {
            border-color: #667eea;
            background-color: #f0f0ff;
        }
        
        .file-input {
            display: none;
        }
        
        .upload-icon {
            font-size: 3rem;
            color: #667eea;
            margin-bottom: 1rem;
        }
        
        .upload-text {
            color: #666;
            margin-bottom: 0.5rem;
        }
        
        .file-info {
            margin-top: 1rem;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            display: none;
        }
        
        .analyze-btn {
            width: 100%;
            padding: 0.75rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
            margin-top: 1rem;
            display: none;
        }
        
        .analyze-btn:hover {
            transform: translateY(-2px);
        }
        
        .analyze-btn:disabled {
            opacity: 0.7;
            cursor: not-allowed;
        }
        
        .progress {
            display: none;
            margin-top: 1rem;
        }
        
        .progress-bar {
            width: 100%;
            height: 4px;
            background-color: #f0f0f0;
            border-radius: 2px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .status-message {
            margin-top: 0.5rem;
            text-align: center;
            color: #666;
            font-size: 0.9rem;
        }
        
        .results {
            display: none;
            margin-top: 2rem;
        }
        
        .result-section {
            margin-bottom: 1.5rem;
        }
        
        .result-title {
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .result-content {
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        
        .download-buttons {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .download-btn {
            flex: 1;
            padding: 0.5rem;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 0.5rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
            text-align: center;
        }
        
        .download-btn:hover {
            background: #667eea;
            color: white;
        }
        
        .error {
            color: #dc3545;
            margin-top: 1rem;
            padding: 1rem;
            background-color: #f8d7da;
            border-radius: 0.5rem;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Executive Summary Analyzer</h1>
        <p class="subtitle">Upload a PDF document to analyze company information</p>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📄</div>
            <p class="upload-text">Drag and drop your PDF here or click to browse</p>
            <input type="file" class="file-input" id="fileInput" accept=".pdf">
        </div>
        
        <div class="file-info" id="fileInfo">
            <strong>Selected file:</strong> <span id="fileName"></span>
        </div>
        
        <button class="analyze-btn" id="analyzeBtn">Analyze Document</button>
        
        <div class="progress" id="progress">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <p class="status-message" id="statusMessage">Processing...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="results" id="results">
            <h2>Analysis Results</h2>
            <div id="resultContent"></div>
            <div class="download-buttons" id="downloadButtons"></div>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const progress = document.getElementById('progress');
        const progressFill = document.getElementById('progressFill');
        const statusMessage = document.getElementById('statusMessage');
        const error = document.getElementById('error');
        const results = document.getElementById('results');
        const resultContent = document.getElementById('resultContent');
        const downloadButtons = document.getElementById('downloadButtons');
        
        let selectedFile = null;
        
        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        // File selection
        fileInput.addEventListener('change', (e) => {
            handleFileSelection(e.target.files[0]);
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragging');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragging');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragging');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelection(files[0]);
            }
        });
        
        function handleFileSelection(file) {
            if (!file) return;
            
            if (file.type !== 'application/pdf') {
                showError('Please select a PDF file');
                return;
            }
            
            selectedFile = file;
            fileName.textContent = file.name;
            fileInfo.style.display = 'block';
            analyzeBtn.style.display = 'block';
            error.style.display = 'none';
            results.style.display = 'none';
        }
        
        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            analyzeBtn.disabled = true;
            progress.style.display = 'block';
            error.style.display = 'none';
            results.style.display = 'none';
            
            // Simulate progress
            let progressValue = 0;
            const progressInterval = setInterval(() => {
                progressValue += Math.random() * 15;
                if (progressValue > 90) progressValue = 90;
                progressFill.style.width = progressValue + '%';
            }, 500);
            
            try {
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                statusMessage.textContent = 'Uploading document...';
                
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                clearInterval(progressInterval);
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Analysis failed');
                }
                
                statusMessage.textContent = 'Analysis complete!';
                progressFill.style.width = '100%';
                
                const data = await response.json();
                displayResults(data);
                
            } catch (err) {
                clearInterval(progressInterval);
                showError(err.message);
            } finally {
                analyzeBtn.disabled = false;
                setTimeout(() => {
                    progress.style.display = 'none';
                    progressFill.style.width = '0%';
                }, 1000);
            }
        });
        
        function displayResults(data) {
            results.style.display = 'block';
            
            // Display extracted information
            const info = data.extracted_info;
            let infoHtml = '<div class="result-section"><h3>Extracted Information</h3><div class="result-content">';
            
            const fields = [
                { key: 'company_name', label: 'Company Name' },
                { key: 'technology_type', label: 'Technology Type' },
                { key: 'product_development_stage', label: 'Development Stage' },
                { key: 'market_size', label: 'Market Size' },
                { key: 'investment_needed', label: 'Investment Needed' },
                { key: 'years_to_exit', label: 'Years to Exit' }
            ];
            
            fields.forEach(field => {
                const value = info[field.key] || 'Not provided';
                const confidence = info.confidence_scores[field.key] || 0;
                infoHtml += `<p><strong>${field.label}:</strong> ${value} 
                    <span style="color: #666; font-size: 0.8em;">(Confidence: ${(confidence * 100).toFixed(0)}%)</span></p>`;
            });
            
            infoHtml += '</div></div>';
            
            // Display report sections
            const report = data.report;
            infoHtml += '<div class="result-section"><h3>Analysis Report</h3>';
            
            const sections = [
                { key: 'nature_and_state', label: 'Nature and State of the Product' },
                { key: 'market_need_and_size', label: 'Market Need and Size' },
                { key: 'roi_elements', label: 'Elements of Potential ROI' },
                { key: 'management_team_strength', label: 'Strength of the Management Team' }
            ];
            
            sections.forEach(section => {
                infoHtml += `<div class="result-section">
                    <div class="result-title">${section.label}</div>
                    <div class="result-content">${report[section.key] || 'No information available'}</div>
                </div>`;
            });
            
            resultContent.innerHTML = infoHtml;
            
            // Display download buttons
            downloadButtons.innerHTML = `
                <a href="/api/report/html/${data.report_files.html.replace('reports/', '')}" 
                   class="download-btn" download>Download HTML Report</a>
                <a href="/api/report/markdown/${data.report_files.markdown.replace('reports/', '')}" 
                   class="download-btn" download>Download Markdown Report</a>
            `;
        }
        
        function showError(message) {
            error.textContent = 'Error: ' + message;
            error.style.display = 'block';
        }
    </script>
</body>
</html>"""
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)

# Environment setup script
def create_setup_files():
    """Create setup and configuration files"""
    
    # Create requirements.txt
    requirements = """flask==2.3.3
flask-cors==4.0.0
werkzeug==2.3.7
PyPDF2==3.0.1
pdfplumber==0.10.2
openai==0.28.0
langchain==0.0.310
python-dotenv==1.0.0
markdown==3.4.4
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    # Create .env.example
    env_example = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Flask Configuration
SECRET_KEY=your_secret_key_here
FLASK_ENV=development
FLASK_DEBUG=True

# Server Configuration
HOST=127.0.0.1
PORT=5000
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_example)
    
    # Create README.md
    readme = """# Executive Summary Analyzer

A comprehensive web application for analyzing company executive summaries using Large Language Models (LLMs).

## Features

- **PDF Upload**: Drag-and-drop or click to upload PDF documents
- **AI-Powered Analysis**: Extract key information using OpenAI GPT-4
- **Structured Reports**: Generate professional investment analysis reports
- **Multiple Export Formats**: Download reports as HTML or Markdown
- **Confidence Scoring**: Each extracted field includes confidence scores
- **Robust PDF Processing**: Handles various PDF formats and layouts

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd executive-summary-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `SECRET_KEY`: Flask secret key for session management
- `FLASK_ENV`: Set to 'development' or 'production'
- `HOST`: Server host (default: 127.0.0.1)
- `PORT`: Server port (default: 5000)

## API Endpoints

- `GET /`: Main web interface
- `POST /api/analyze`: Upload and analyze PDF document
- `GET /api/report/<format>/<filename>`: Download generated report
- `GET /health`: Health check endpoint

## Project Structure

```
executive-summary-analyzer/
├── app.py                 # Main application file
├── templates/            
│   └── index.html        # Web interface
├── uploads/              # Temporary file storage
├── reports/              # Generated reports
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## Usage

1. Open the web interface
2. Upload a PDF executive summary
3. Click "Analyze Document"
4. View the extracted information and generated report
5. Download the report in HTML or Markdown format

## Error Handling

The application includes comprehensive error handling for:
- Invalid file types
- PDF extraction failures
- API errors
- Network issues

## Security Considerations

- File uploads are validated and sanitized
- Temporary files are cleaned up after processing
- API keys are stored securely in environment variables
- CORS is configured for API endpoints

## License

This project is provided as-is for educational and commercial use.
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    
    # Create run.py for easy startup
    run_script = """#!/usr/bin/env python
import os
from app import app, create_templates, create_setup_files

if __name__ == '__main__':
    # Create necessary directories and files
    create_templates()
    
    # Get configuration from environment
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"Starting Executive Summary Analyzer...")
    print(f"Server running at http://{host}:{port}")
    print(f"Debug mode: {debug}")
    
    # Run the application
    app.run(host=host, port=port, debug=debug)
"""
    
    with open('run.py', 'w') as f:
        f.write(run_script)
    
    # Make run.py executable on Unix systems
    try:
        os.chmod('run.py', 0o755)
    except: #noqa
        pass

# Main execution
if __name__ == '__main__':
    # Setup the application
    create_templates()
    create_setup_files()
    
    # Run the application
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"Executive Summary Analyzer starting...") #noqa
    print(f"Server running at http://{host}:{port}")
    
    app.run(host=host, port=port, debug=debug)