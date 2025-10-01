from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Step 1: Load and process PDF data
extracted_data = load_pdf_files(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# Step 2: Load embeddings
embeddings = download_embeddings()

# Step 3: Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Step 4: Store documents in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
