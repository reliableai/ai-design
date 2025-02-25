# Markdown Survey Creator

A simple web application that converts markdown files into interactive online surveys. Create surveys using an easy-to-write markdown format and collect responses in a structured way.

## Features

- Convert markdown files to interactive web forms
- Support for various question types:
  - Text input
  - Text area
  - Radio buttons
  - Checkboxes
  - Dropdown select
  - Rating scales (1-5)
- Required field validation
- Section organization
- Response storage in JSON format
- Survey metadata support
- Real-time preview
- Mobile-friendly interface

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   cd src
   streamlit run app.py
   ```

2. Create a markdown survey file following the format below
3. Upload your survey file through the web interface
4. Share the URL with respondents
5. Collect responses in the `data/responses` directory

## Markdown Format

Surveys are written in markdown with a specific format:

```markdown
---
title: Survey Title
description: Survey description
created_by: Author Name
response_folder: folder_name
---

# Survey Title

Survey introduction text

## Section Name

### Question Text *
[text: question_id required]

### Multiple Choice Question
[radio: question_id]
- Option 1
- Option 2
- Option 3

### Dropdown Question
[select: question_id]
- Choice 1
- Choice 2
- Choice 3

### Checkbox Question
[checkbox: question_id]
- Item 1
- Item 2
- Item 3

### Scale Questions
[scale: topic_ratings]
- Topic 1 (1-5)
- Topic 2 (1-5)
- Topic 3 (1-5)

### Long Answer
[textarea: feedback]
```

### Question Types

- `[text: id]`: Single line text input
- `[textarea: id]`: Multi-line text input
- `[radio: id]`: Radio button selection
- `[select: id]`: Dropdown selection
- `[checkbox: id]`: Multiple choice checkboxes
- `[scale: id]`: 1-5 rating scales

Add `required` to make a field mandatory.

## Response Storage

Responses are stored as JSON files in the `data/responses` directory with the following format:

```json
{
  "timestamp": "2024-03-15T10:30:00",
  "survey_title": "Survey Title",
  "responses": {
    "question_id": "response_value",
    "checkbox_id": ["selected", "items"],
    "scale_id": {
      "Topic 1": 4,
      "Topic 2": 3
    }
  }
}
```

## Sample Survey

A sample survey is included in `data/sample_survey.md`. You can use this as a template for creating your own surveys.

## Notes

- Responses are stored locally in JSON format
- Each response file is named with the survey title and timestamp
- The application is built with Streamlit and can be easily deployed to Streamlit Cloud
- The interface is responsive and works well on mobile devices 