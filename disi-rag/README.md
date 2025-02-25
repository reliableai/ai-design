# DISI Schedule Assistant

A RAG-based chatbot that helps students find information about schedules and locations at DISI (Department of Information Engineering and Computer Science) at the University of Trento.

## Features

- Web scraping of DISI website content
- Local LLM integration using Ollama
- Conversational interface with memory
- Source tracking for answers
- Easy-to-use Streamlit interface

## Prerequisites

- Python 3.8+
- Ollama installed with llama2 model
  - Install Ollama from: https://ollama.ai/
  - Pull the model: `ollama pull llama2`

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

2. In the web interface:
   - The sidebar contains configuration options
   - Set the base URL (default is https://www.disi.unitn.it)
   - Set the maximum number of pages to scrape
   - Click "Initialize System" to start
   - Begin chatting in the main interface

## How it Works

1. The WebScraper class crawls the DISI website and extracts text content
2. The RAGSystem class:
   - Splits the content into chunks
   - Creates embeddings using Ollama
   - Stores them in a Chroma vector database
   - Uses LangChain for retrieval and response generation

## Notes

- The system maintains conversation history
- Sources are tracked for each response
- The local LLM ensures privacy and faster responses
- Web scraping is done only during initialization 