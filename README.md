# LexEval

A simple tool for testing legal language models. Currently supports Together.ai models for tasks like license understanding, clause analysis, or anything else that reflects your legal work.

## Requirements

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Quick Start

1. Clone and install:
```bash
git clone https://github.com/ryanmcdonough/lexeval.git
cd lexeval
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Get your Together.ai API key from [Together.ai](https://www.together.ai)

## Usage

1. Make sure your virtual environment is activated:
```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Start the app:
```bash
streamlit run frontend/streamlit_app.py
```

3. Open http://localhost:8501 in your browser

4. Enter your API key and select a model/task to evaluate

5. Click "Run Evaluation" to start

## What's Inside

```
lexeval/
├── frontend/          # Web interface
├── tasks/            # Your evaluation tasks
├── results/          # Where results get saved
├── core/             # Main code
└── config/           # Configuration files
```

## Creating Tasks

Tasks are stored as JSON files in the `tasks/` folder. Here's a basic example:

```json
{
    "schema_version": "1.0",
    "task_name": "Your Task Name",
    "description": "Description of your task",
    "created_at": 1234567890,
    "tasks": [
        {
            "task_id": "unique_id",
            "task_name": "Task Name",
            "prompt": "Your prompt text",
            "context": "Context for the task",
            "expected_output": "Expected model output",
            "reference": "Reference answer",
            "metric": "rouge|keyword_match|llm_judge|human_review"
        }
    ]
}
```

Available metrics:
- `keyword_match`: Checks for key terms
- `rouge`: Text similarity scoring
- `llm_judge`: Uses another model to evaluate
- `human_review`: Manual review needed

## Notes

- Results are saved in both CSV and JSON formats
- You can use the web interface to create/edit tasks
- Default models are included if the API is unavailable

## Troubleshooting

If you see dependency conflicts during installation:
1. Create a fresh virtual environment
2. Install dependencies in this order:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## License

MIT License