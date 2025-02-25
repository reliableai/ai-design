import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class WebScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.visited_urls = set()

    def get_domain(self, url: str) -> str:
        return '/'.join(url.split('/')[:3])

    def is_valid_url(self, url: str) -> bool:
        domain = self.get_domain(self.base_url)
        return url.startswith(domain)

    def scrape_page(self, url: str) -> Dict[str, str]:
        if url in self.visited_urls:
            return {}
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                return {}
            
            self.visited_urls.add(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return {url: text}
            
        except Exception as e:
            st.error(f"Error scraping {url}: {str(e)}")
            return {}

    def get_links(self, url: str) -> List[str]:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    if href.startswith('/'):
                        href = self.get_domain(self.base_url) + href
                    if self.is_valid_url(href):
                        links.append(href)
            
            return list(set(links))
        except:
            return []

    def scrape_domain(self, max_pages: int = 50) -> Dict[str, str]:
        to_visit = [self.base_url]
        content = {}
        
        while to_visit and len(content) < max_pages:
            url = to_visit.pop(0)
            page_content = self.scrape_page(url)
            content.update(page_content)
            
            if len(content) < max_pages:
                new_links = self.get_links(url)
                to_visit.extend([link for link in new_links if link not in self.visited_urls])
        
        return content

class RAGSystem:
    def __init__(self, model_name: str = "llama2"):
        self.llm = Ollama(model=model_name)
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.vectorstore = None
        self.chain = None

    def ingest_texts(self, texts: Dict[str, str]):
        documents = []
        for url, text in texts.items():
            chunks = self.text_splitter.split_text(text)
            documents.extend([f"Source: {url}\n\n{chunk}" for chunk in chunks])
        
        self.vectorstore = Chroma.from_texts(
            documents,
            self.embeddings,
            collection_name="disi_docs"
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )

    def query(self, question: str) -> Dict:
        if not self.chain:
            return {"answer": "Please ingest some documents first."}
        
        return self.chain({"question": question})

def main():
    st.title("DISI Schedule Assistant")
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
        st.session_state.is_initialized = False

    with st.sidebar:
        st.header("Configuration")
        base_url = st.text_input(
            "Base URL",
            value="https://www.disi.unitn.it",
            help="Enter the base URL to scrape"
        )
        max_pages = st.number_input(
            "Max Pages to Scrape",
            min_value=1,
            max_value=100,
            value=20
        )
        
        if st.button("Initialize System"):
            with st.spinner("Scraping website and initializing RAG system..."):
                scraper = WebScraper(base_url)
                texts = scraper.scrape_domain(max_pages)
                st.session_state.rag_system.ingest_texts(texts)
                st.session_state.is_initialized = True
                st.success("System initialized!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about DISI schedules and locations"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.is_initialized:
            with st.chat_message("assistant"):
                error_msg = "Please initialize the system first using the sidebar!"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.markdown(error_msg)
        else:
            with st.chat_message("assistant"):
                response = st.session_state.rag_system.query(prompt)
                answer = response["answer"]
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.markdown(answer)

if __name__ == "__main__":
    main() 