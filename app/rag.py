import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self, cv_path=None):
        self.cv_path = cv_path or os.path.join(os.path.dirname(__file__), "cv_data.txt")
        self.vector_store = None
        self.rag_chain = None

    def initialize_rag(self):
        if self.rag_chain:
            return  # already initialized

        if not os.path.exists(self.cv_path):
            error_msg = f"CV data file not found at {self.cv_path}. Please ensure 'app/cv_data.txt' exists."
            print(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)

        if not os.getenv("OPENAI_API_KEY"):
            error_msg = "OPENAI_API_KEY not found in environment variables. Check Render dashboard."
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        try:
            print(f"Loading data from {self.cv_path}...")
            loader = TextLoader(self.cv_path)
            documents = loader.load()

            print("Splitting text into chunks...")
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            print("Creating embeddings and vector store (FAISS)...")
            embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.from_documents(texts, embeddings)

            print("Initializing modern RAG chain...")
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            
            # Define system prompt
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Use three sentences maximum and keep the answer concise."
                "\n\n"
                "{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # Create document chain
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            
            # Create retrieval chain
            self.rag_chain = create_retrieval_chain(
                self.vector_store.as_retriever(),
                question_answer_chain
            )

            print("RAG System initialization complete.")
        except Exception as e:
            full_error = f"Error during RAG initialization: {str(e)}"
            print(f"CRITICAL ERROR: {full_error}")
            raise RuntimeError(full_error)

    def query(self, question: str):
        if not self.rag_chain:
            self.initialize_rag()

        try:
            # Use 'invoke' with 'input' key as required by create_retrieval_chain
            response = self.rag_chain.invoke({"input": question})
            return response.get("answer", "No answer found.")
        except Exception as e:
            return f"Error processing query: {str(e)}"

# Singleton instance
rag_system = RAGSystem()
