import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self, cv_path="cv_data.txt"):
        self.cv_path = cv_path
        self.vector_store = None
        self.qa_chain = None
        self.initialize_rag()

    def initialize_rag(self):
        if not os.path.exists(self.cv_path):
            print(f"Warning: {self.cv_path} not found.")
            return

        # Check for API Key
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            return

        try:
            # 1. Load Data
            loader = TextLoader(self.cv_path)
            documents = loader.load()

            # 2. Split Text
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            # 3. Create Embeddings & Vector Store
            embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.from_documents(texts, embeddings)

            # 4. Create Retrieval Chain
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever()
            )
            print("RAG System initialized successfully.")
        except Exception as e:
            print(f"Error initializing RAG: {e}")

    def query(self, question: str):
        if not self.qa_chain:
            return "System not initialized or API key missing."
        
        try:
            response = self.qa_chain.invoke(question)
            return response.get("result", "No answer found.")
        except Exception as e:
            return f"Error processing query: {str(e)}"

# Singleton instance
rag_system = RAGSystem()
