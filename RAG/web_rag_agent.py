import os
import requests
import tempfile
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import BSHTMLLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


class WebRAGAgent:
    """
    An agent that scrapes a website, builds a local vector knowledge base,
    and answers user queries using RAG (Retrieval-Augmented Generation).
    """

    def __init__(self,
                 model_name: str = "gpt-4o-mini",
                 chunk_size: int = 300,
                 chunk_overlap: int = 50,
                 max_tokens: int = 15000,
                 temperature: float = 0.4,
                 output_dir: str = "outputs"):
        """
        Initializes the RAG Agent with configuration parameters.
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Setup output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(output_dir, f"{timestamp}_web_rag")
        os.makedirs(self.run_dir, exist_ok=True)
        self.chat_log_path = os.path.join(self.run_dir, "chat_history.txt")

        # Components
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None

        # API Key Check
        self._check_api_key()
        self._init_llm()

    def _check_api_key(self):
        """Ensures OpenAI API key is set."""
        if not os.environ.get("OPENAI_API_KEY"):
            key = input("\nPlease enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = key

    def _init_llm(self):
        """Initializes the Language Model and Memory."""
        print(f"Initializing LLM: {self.model_name}...")
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def fetch_url_content(self, url: str) -> List[Any]:
        """
        Fetches and parses HTML content from a URL into Document objects.
        """
        print(f"\nFetching content from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch URL: {e}")
            return []

        # Use temp file for BSHTMLLoader
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as tmp:
            tmp.write(response.text)
            tmp_path = tmp.name

        try:
            # Try loading with default parser
            loader = BSHTMLLoader(tmp_path)
            documents = loader.load()
        except ImportError:
            print("'lxml' not found. Falling back to 'html.parser'.")
            loader = BSHTMLLoader(tmp_path, bs_kwargs={'features': 'html.parser'})
            documents = loader.load()
        finally:
            os.unlink(tmp_path)

        return documents

    def process_documents(self, documents: List[Any]):
        """
        Splits documents into chunks and builds the Vector Store.
        """
        if not documents:
            raise ValueError("No documents to process.")

        print(f"Loaded {len(documents)} document(s).")

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]  # Try to split by paragraph, then line, then word
        )
        texts = text_splitter.split_documents(documents)
        print(f"Created {len(texts)} text chunks.")

        # Create Embeddings & Vector Store
        print("Generating Embeddings and building Vector Store (FAISS)...")
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(texts, embeddings)

        # Verify with a sample
        self._print_sample_embedding(texts[0].page_content, embeddings)

        # Setup Retrieval Chain
        self._setup_qa_chain()

    def _print_sample_embedding(self, text: str, embedding_model):
        """Prints a debug sample of the embedding process."""
        vec = embedding_model.embed_query(text)
        print("\n" + "-" * 40)
        print(f"Sample Embedding (First 5 dims): {np.array(vec[:5])}")
        print(f"Embedding Shape: {len(vec)}")
        print("-" * 40 + "\n")

    def _setup_qa_chain(self):
        """Configures the RetrievalQA chain with a custom prompt."""
        template = """Context: {context}

Question: {question}

Answer the question concisely based only on the given context. 
If the context doesn't contain relevant information, say "I don't have enough information."
However, if the question is generic (e.g. "what is AI?"), you may answer using general knowledge.
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            chain_type_kwargs={"prompt": prompt}
        )
        print("RAG Pipeline is ready.")

    def query(self, question: str) -> str:
        """Runs the RAG pipeline for a single question."""
        if not self.qa_chain:
            return "Error: Pipeline not initialized. Please load a URL first."

        # Debug: Show retrieved chunks
        relevant_docs = self.vectorstore.similarity_search_with_score(question, k=3)
        print(f"\nFound {len(relevant_docs)} relevant chunks.")

        response = self.qa_chain.invoke({"query": question})
        return response['result']

    def run_interactive_session(self):
        """Runs the chat session """
        print("\n" + "=" * 50)
        print(" WELCOME TO WEB-RAG AGENT (University of York Edition)")
        print("=" * 50)

        # --- URL HERE ---
        target_url = "https://en.wikipedia.org/wiki/University_of_York"
        print(f"\nAuto-loading University of York URL: {target_url}")

        # 1. Load and Process automatically
        docs = self.fetch_url_content(target_url)
        if not docs:
            print("Could not load the hardcoded URL. Exiting.")
            return

        try:
            self.process_documents(docs)
        except Exception as e:
            print(f"Processing failed: {e}")
            return

        # 2. Chat Loop
        print("\n" + "-" * 50)
        print("Knowledge Base Ready! Ask questions about University of York.")
        print("Type 'quit' to exit.")
        print("-" * 50 + "\n")

        with open(self.chat_log_path, "w", encoding="utf-8") as f:
            f.write(f"Chat Session Started: {datetime.now()}\n")
            f.write(f"Source URL: {target_url}\n")
            f.write("=" * 50 + "\n\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break

            # Simple check to prevent empty inputs
            if not user_input:
                continue

            response = self.query(user_input)
            print(f"Agent: {response}")

            with open(self.chat_log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"You: {user_input}\n")
                log_file.write(f"Agent: {response}\n")
                log_file.write("-" * 30 + "\n")


# --- Main ---
if __name__ == "__main__":
    # Initialize the Agent
    agent = WebRAGAgent(
        model_name="gpt-4o-mini",
        chunk_size=300,
        chunk_overlap=50
    )

    # Start the application
    agent.run_interactive_session()