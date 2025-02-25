import streamlit as st
import frontmatter
import yaml
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import re
import os

class SurveyParser:
    def __init__(self, markdown_content):
        # Parse frontmatter and content
        self.survey = frontmatter.loads(markdown_content)
        self.title = self.survey.metadata.get('title', 'Survey')
        self.description = self.survey.metadata.get('description', '')
        self.created_by = self.survey.metadata.get('created_by', '')
        self.response_folder = self.survey.metadata.get('response_folder', 'responses')
        self.content = self.survey.content

    def parse_question(self, question_text):
        """Extract question type and properties from markdown syntax."""
        match = re.search(r'\[(.*?):(.*?)(?: required)?\]', question_text)
        if not match:
            return None
        
        q_type, q_id = match.groups()
        required = 'required' in question_text
        
        return {
            'type': q_type.strip(),
            'id': q_id.strip(),
            'required': required
        }

    def get_options(self, lines, start_idx):
        """Extract options for radio, checkbox, select, or scale questions."""
        options = []
        i = start_idx
        while i < len(lines) and lines[i].startswith('- '):
            option = lines[i][2:].strip()
            if '(1-5)' in option:  # Handle scale questions
                option = option.replace(' (1-5)', '')
            options.append(option)
            i += 1
        return options, i

    def parse_survey(self):
        """Parse the markdown content into a structured survey format."""
        lines = self.content.split('\n')
        questions = []
        current_section = None
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            
            # Handle sections
            if line.startswith('## '):
                current_section = line[3:].strip()
            
            # Handle questions
            elif line.startswith('### '):
                question_text = line[4:].strip()
                i += 1
                if i < len(lines):
                    question_def = self.parse_question(lines[i])
                    if question_def:
                        question = {
                            'section': current_section,
                            'text': question_text.replace(' *', ''),
                            'required': question_def['required'],
                            **question_def
                        }
                        
                        # Get options for certain question types
                        if question['type'] in ['radio', 'checkbox', 'select', 'scale']:
                            i += 1
                            options, i = self.get_options(lines, i)
                            question['options'] = options
                            i -= 1
                        
                        questions.append(question)
            i += 1
        
        return questions

def save_response(response_data, survey_title):
    """Save survey response to a JSON file."""
    # Create responses directory if it doesn't exist
    response_dir = Path("data/responses")
    response_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{survey_title.lower().replace(' ', '_')}_{timestamp}.json"
    
    # Save response
    with open(response_dir / filename, 'w') as f:
        json.dump(response_data, f, indent=2)

def create_survey_form(questions):
    """Create Streamlit form elements based on question type."""
    responses = {}
    
    current_section = None
    for question in questions:
        # Handle sections
        if question['section'] != current_section:
            current_section = question['section']
            st.subheader(current_section)
        
        # Add required indicator
        label = f"{question['text']}{'*' if question['required'] else ''}"
        
        # Create form element based on question type
        if question['type'] == 'text':
            response = st.text_input(label, key=question['id'])
        
        elif question['type'] == 'textarea':
            response = st.text_area(label, key=question['id'])
        
        elif question['type'] == 'radio':
            response = st.radio(label, question['options'], key=question['id'])
        
        elif question['type'] == 'select':
            response = st.selectbox(label, question['options'], key=question['id'])
        
        elif question['type'] == 'checkbox':
            response = st.multiselect(label, question['options'], key=question['id'])
        
        elif question['type'] == 'scale':
            st.write(label)
            scale_responses = {}
            for option in question['options']:
                scale_responses[option] = st.slider(
                    option, 
                    min_value=1, 
                    max_value=5, 
                    value=3, 
                    key=f"{question['id']}_{option}"
                )
            response = scale_responses
        
        responses[question['id']] = response
    
    return responses

def main():
    st.title("Markdown Survey Creator")
    
    # File upload section in sidebar
    with st.sidebar:
        st.header("Survey Configuration")
        uploaded_file = st.file_uploader("Upload Survey Markdown", type=['md'])
        
        if uploaded_file:
            survey_content = uploaded_file.read().decode()
            st.session_state.survey_parser = SurveyParser(survey_content)
            st.session_state.questions = st.session_state.survey_parser.parse_survey()
    
    # Display sample format if no file is uploaded
    if 'survey_parser' not in st.session_state:
        st.info("Please upload a survey markdown file to begin.")
        with st.expander("View Sample Survey Format"):
            with open('data/sample_survey.md', 'r') as f:
                st.code(f.read(), language='markdown')
        return
    
    # Display survey
    st.title(st.session_state.survey_parser.title)
    st.write(st.session_state.survey_parser.description)
    
    # Create form
    with st.form("survey_form"):
        responses = create_survey_form(st.session_state.questions)
        submit_button = st.form_submit_button("Submit Survey")
        
        if submit_button:
            # Validate required fields
            missing_required = [q['text'] for q in st.session_state.questions 
                              if q['required'] and not responses[q['id']]]
            
            if missing_required:
                st.error("Please fill in all required fields:\n" + 
                        "\n".join(f"- {field}" for field in missing_required))
            else:
                # Add metadata to response
                response_data = {
                    'timestamp': datetime.now().isoformat(),
                    'survey_title': st.session_state.survey_parser.title,
                    'responses': responses
                }
                
                # Save response
                save_response(response_data, st.session_state.survey_parser.title)
                st.success("Thank you for completing the survey!")
                st.balloons()

if __name__ == "__main__":
    main() 