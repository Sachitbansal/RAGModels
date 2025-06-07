from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import faiss
import json
from sentence_transformers import SentenceTransformer # Added for FAISS embeddings

load_dotenv() # Load environment variables, including your GOOGLE_API_KEY

class LocalRAGSystemFAISS: # Changed class name to reflect FAISS
    def __init__(self, llm_model_name: str = "gemini-2.5-flash-preview-04-17"):
        """
        Initializes the RAG system with SentenceTransformer for embeddings (for FAISS),
        FAISS for vector search, and Google Generative AI (Gemini Flash) for the LLM.
        
        Args:
            llm_model_name: The name of the Google Generative AI LLM model to use.
                            Defaults to "gemini-2.5-flash-preview-04-17".
        """
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # SentenceTransformer for FAISS
        self.llm = GoogleGenerativeAI(model=llm_model_name)
        self.faiss_index = None # Will hold the FAISS index
        self.metadata = None    # Will hold the associated text content (from meta.json)
        self.qa_chain = None

    def _create_and_save_faiss_index(self, file_path: str, faiss_index_path: str, meta_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Helper function to create embeddings, build a FAISS index, and save the index and metadata.
        This runs if the FAISS index or metadata files aren't found.
        """
        print(f"Creating FAISS index and metadata from: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        raw_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(raw_documents)

        texts = [doc.page_content for doc in docs]
        
        print("Generating embeddings with SentenceTransformer...")
        embeddings = self.embedding_model.encode(texts).astype('float32')

        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension) # Using L2 (Euclidean) distance for similarity
        faiss_index.add(embeddings)
        print(f"FAISS index created with {faiss_index.ntotal} vectors.")

        faiss.write_index(faiss_index, faiss_index_path)
        print(f"FAISS index saved to {faiss_index_path}")

        # Prepare metadata (list of dictionaries, each with 'text_preview' and potentially other info)
        metadata_list = [{"text_preview": text, "source": doc.metadata.get('source', 'N/A')} for text, doc in zip(texts, docs)]
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": metadata_list}, f, indent=4) # Wrap in "chunks" key for consistency
        print(f"Metadata saved to {meta_path}")
        return faiss_index, {"chunks": metadata_list}

    def load_faiss_index_and_metadata(self, faiss_index_path: str = "faiss_index.idx", meta_path: str = "meta.json", file_path: str = "fold/common.txt"):
        """
        Loads the pre-existing FAISS index and metadata. If files don't exist,
        it will attempt to create them from the provided text file.
        
        Args:
            faiss_index_path: Path to the saved FAISS index file.
            meta_path: Path to the saved JSON metadata file.
            file_path: Path to the original text file, used if index/meta need creation.
        """
        if not os.path.exists(faiss_index_path) or not os.path.exists(meta_path):
            print("FAISS index or metadata not found. Creating them now...")
            self.faiss_index, self.metadata = self._create_and_save_faiss_index(file_path, faiss_index_path, meta_path)
        else:
            print(f"Loading FAISS index from {faiss_index_path}...")
            self.faiss_index = faiss.read_index(faiss_index_path)
            print(f"Loading metadata from {meta_path}...")
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print("FAISS index and metadata loaded successfully.")

    def setup_qa_chain_with_prompt(self, custom_prompt_template: str):
        """
        Configures the LLMChain with your specified prompt template. This chain will take
        the user's question and retrieved documents to generate an answer.
        
        Args:
            custom_prompt_template: The string template for the prompt.
                                   It *must* contain '{question}' and '{docs}' placeholders.
        """
        if self.faiss_index is None or self.metadata is None:
            raise ValueError("FAISS index or metadata not loaded. Please call `load_faiss_index_and_metadata` first.")

        prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template=custom_prompt_template,
        )

        self.qa_chain = LLMChain(llm=self.llm, prompt=prompt)
        print("QA chain setup complete with custom prompt.")

    def get_response_from_query(self, query: str, k: int = 4) -> tuple[str, list]:
        """
        Retrieves the most similar documents using FAISS and uses them as context
        for the LLM to answer the user's question.
        
        Args:
            query: The question you want to ask your RAG system.
            k: The number of top relevant documents to retrieve from FAISS.
        Returns:
            A tuple containing:
            - The LLM's generated answer (as a string).
            - A list of the source document dictionaries (from metadata) that were used for context.
        """
        if self.faiss_index is None or self.metadata is None:
            raise ValueError("FAISS index or metadata not loaded. Please call `load_faiss_index_and_metadata` first.")
        if self.qa_chain is None:
            raise ValueError("QA chain not set up. Please call `setup_qa_chain_with_prompt` first.")

        print(f"Processing query: \"{query}\"")

        # Encode the query using the same SentenceTransformer model
        query_embedding = self.embedding_model.encode([query]).astype("float32")
        
        # Perform similarity search using FAISS
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Removed print(docs) and print("sachitbansaldocs")
        
        # Retrieve the original text content from metadata
        retrieved_docs_data = []
        for i in indices[0]:
            if i < len(self.metadata["chunks"]): # Safety check
                retrieved_docs_data.append(self.metadata["chunks"][i])
        
        # Combine the text previews of retrieved documents into a single string
        docs_page_content = " ".join([d["text_preview"] for d in retrieved_docs_data])

        # Run the LLMChain with the combined context and the question
        response = self.qa_chain.run(question=query, docs=docs_page_content)
        response = response.replace("\n", " ").strip() 
        print("Query processed.")
        return response, retrieved_docs_data # Return the dicts from metadata

if __name__ == "__main__":

    # --- Step 1: Initialize your RAG system ---
    # This sets up the SentenceTransformer for embeddings and Gemini LLM.
    print("\n--- Initializing RAG System ---")
    rag_system = LocalRAGSystemFAISS(llm_model_name="gemini-2.5-flash-preview-04-17")

    # --- Step 2: Load (or create) the FAISS index and metadata ---
    # This assumes 'faiss_index.idx' and 'meta.json' are in the current directory.
    # If not found, it will create them from 'fold/common.txt'.
    print("\n--- Preparing Vector Database (FAISS) ---")
    rag_system.load_faiss_index_and_metadata(
        faiss_index_path="faiss_index.idx",
        meta_path="meta.json",
        file_path="fold/common.txt" # Used only if index/meta files don't exist
    )

    # --- Step 3: Define your specific prompt template ---
    # This template guides Gemini on how to use the retrieved context.
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

    # --- Step 4: Setup the QA chain with your custom prompt ---
    # This prepares the LLM to generate answers.
    print("\n--- Setting up QA Chain ---")
    rag_system.setup_qa_chain_with_prompt(custom_whatsapp_prompt_template)

    # --- Step 5: Define your query ---
    user_query = "Tell me about VLSI branch in IIT Mandi and comparison with Electrical branch?"

    # --- Step 6: Get the final response ---
    print("\n--- Getting Response ---")
    final_response, source_documents = rag_system.get_response_from_query(user_query)

    # --- Step 7: Print the final response and source documents ---
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print(f"Question: {user_query}")
    print(f"Answer: {final_response}")
    print("="*50)
    
    # Optionally, print the source documents for transparency
    print("\n--- Source Documents Used for Answer (Partial Content) ---")
    if source_documents:
        for i, doc_data in enumerate(source_documents):
            print(f"Doc {i+1}: {doc_data.get('text_preview', 'N/A')}...") # Print first 150 chars
            print(f"  Source: {doc_data.get('source', 'N/A')}")
    else:
        print("No source documents retrieved.")