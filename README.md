# Executive Summary Analyzer

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
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
