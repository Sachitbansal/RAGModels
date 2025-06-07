from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv() # Load environment variables, including your GOOGLE_API_KEY

class LocalRAGSystemGeminiFlash:
    def __init__(self, embedding_model_name: str = "models/embedding-001", llm_model_name: str = "gemini-2.5-flash-preview-04-17"):
  
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
        self.llm = GoogleGenerativeAI(model=llm_model_name)
        self.vectorstore = None # This will hold our Chroma vector store
        self.qa_chain = None

    def create_db_from_text_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, persist_directory: str = "./chroma_db_flash") -> Chroma:

        loader = TextLoader(file_path, encoding="utf-8")
        transcript = loader.load() # Using 'transcript' for consistency with your original example's variable naming

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(transcript)

        # Ensure the persistence directory exists
        os.makedirs(persist_directory, exist_ok=True)

        print(f"Attempting to load Chroma vector store from {persist_directory}...")
        try:
            # Try to load an existing vector store
            self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
            # Perform a small dummy query to ensure the loaded DB is functional
            _ = self.vectorstore.similarity_search("check db", k=1) 
            print("Chroma vector store loaded successfully from disk.")
        except Exception as e:
            # If loading fails (e.g., directory is empty or corrupt), create a new one
            print(f"Chroma vector store not found or could not be loaded ({e}). Creating a new one...")
            self.vectorstore = Chroma.from_documents(documents=docs, embedding=self.embeddings, persist_directory=persist_directory)
            self.vectorstore.persist() # Save the newly created store
            print("Chroma vector store created and persisted.")

        return self.vectorstore

    def setup_qa_chain_with_prompt(self, custom_prompt_template: str):

        if self.vectorstore is None:
            raise ValueError("Vector store not created. Please call `create_db_from_text_file` first.")

        # Create the PromptTemplate using the variables expected by your LLMChain
        prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template=custom_prompt_template,
        )

        self.qa_chain = LLMChain(llm=self.llm, prompt=prompt)

    def get_response_from_query(self, query: str, k: int = 4) -> tuple[str, list]:

        if self.vectorstore is None:
            raise ValueError("Vector store not created. Please call `create_db_from_text_file` first.")
        if self.qa_chain is None:
            raise ValueError("QA chain not set up. Please call `setup_qa_chain_with_prompt` first.")

        # Retrieve relevant documents based on similarity to the query
        docs = self.vectorstore.similarity_search(query, k=k)
        
        print(docs)
        print("sachitbansaldocs")
        
        # Combine the page content of retrieved documents into a single string
        docs_page_content = " ".join([d.page_content for d in docs])

        # Run the LLMChain with the combined context and the question
        response = self.qa_chain.run(question=query, docs=docs_page_content)
        # Clean up the response for cleaner printing
        response = response.replace("\n", " ").strip() 
        return response, docs




if __name__ == "__main__":
   
    # 2. Initialize your RAG system using Gemini 1.5 Flash
    # 'gemini-1.5-flash-latest' is the recommended identifier for Gemini 1.5 Flash.
    rag_system = LocalRAGSystemGeminiFlash(llm_model_name="gemini-2.5-flash-preview-04-17")

    # 3. Create or load the database from your common.txt file
    # The vector store will be saved to (or loaded from) './chroma_db_flash'.
    rag_system.create_db_from_text_file("fold/common.txt", persist_directory="./chroma_db_flash")

    # 4. Define your specific prompt template for the IIT Mandi/JOSAA use case.
    # IMPORTANT: Ensure '{question}' and '{docs}' are used as input variables
    # as expected by the LLMChain's PromptTemplate setup.
    custom_whatsapp_prompt_template = """
    You are a helpful assistant that can answer questions about IIT Mandi and JOSAA counselling
    based on the provided context from a chat transcript.

    Answer the following question: {question}
    By searching the following chat transcript context: {docs}

    Only use the factual information from the transcript to answer the question.
    If you feel like you don't have enough information to answer the question, say "I’m not sure based on current information. Please ask to seniors".
    If the question isn’t about JOSAA or IIT choices, acknowledge that and, if applicable, shadow any negative aspects by focusing on positives from the context.
    Do not sound desperate, but do not defame IIT Mandi at any cost. You must highlight positive aspects of the college.
    Lean slightly in favor of IIT Mandi when comparing colleges, but remain accurate.
    Avoid asterisks (*), markdown syntax, or unnecessary exaggeration.
    Please don't mention Anyone's name (like ujjwal or aryan or word someone). Take it as your opinion.
    Do not compare different branches within the same college unless asked.
    Give its plus points over other colleges whenever possible from the given context.

    Your answers should not be too verbose keep them crisp but inlcude all important detail.
    """

    # 5. Setup the QA chain with your custom prompt
    rag_system.setup_qa_chain_with_prompt(custom_whatsapp_prompt_template)

    # 6. Define your query
    user_query = "General Engineering program"

    # 7. Get the response from the query
    final_response, source_documents = rag_system.get_response_from_query(user_query)

    # 8. Print the final response
    print("\n" + "--- Final Answer ---")
    print(f"Question: {user_query}")
    print(f"Answer: {final_response}")
    
    # Optionally, print the source documents for debugging or transparency
    print("\n--- Source Documents Used ---")
    for i, doc in enumerate(source_documents):
        print(f"Doc {i+1}: {doc.page_content[:200]}...") # Print first 200 chars
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")