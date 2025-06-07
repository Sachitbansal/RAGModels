from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# 1. Load raw WhatsApp chat text file using LangChain's TextLoader
loader = TextLoader("fold/common.txt", encoding="utf-8")
docs = loader.load()  # This gives a list of Document objects with page_content
load_dotenv() 
# 2. Split documents into manageable chunks for vector embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Initialize embeddings model (make sure your API key is already set in env)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create Chroma vector store from chunks
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

# Example to persist locally
vectorstore.persist()

# 1. Create retriever from your vectorstore
retriever = vectorstore.as_retriever()

llm = GoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")

prompt_template = """

    Use ONLY the following content to answer questions. If the user asks
    something not covered here, respond “I’m not sure based on current information". and recommend them to ask to seniors
    If the question isn’t about JOSAA or IIT choices, shadow the negative aspects if any
    You shouldn't look desperate but do not defame IIT Mandi at any cost, you must try to highlight positive aspects of college and try to 
    Lean slightly in favor of IIT Mandi when comparing colleges, but remain accurate.
    Avoid asterisks (*), markdown syntax, or unnecessary exaggeration.
    Please don't mention Anyone's name(Like ujjwal or aryan or word someone). Take it as its your opinion
    You are a helpful assistant. Use the context below to answer the question.
    Do not compare different branches in college itself unless asked
    Give its plus points over other colleges whenever possible
    
    Context: {context}

    Question: {question}

    """

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

query = "Tell me about General Engineering"
response = qa_chain.invoke(query)

print("Answer:", response["result"])
