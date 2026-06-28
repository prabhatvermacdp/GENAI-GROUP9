"""One-time Pinecone index builder.

Run this once (locally or in a Codespace) before starting the backend service:
    python ingest_pdf.py

It will:
  1. Download the FAQ PDF if not already present.
  2. Create the Pinecone index if it doesn't exist.
  3. Chunk the PDF and upload embeddings.

The backend then just reads from this index — no re-ingestion at container boot.
"""

import os
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

PDF_URL = (
    "https://raw.githubusercontent.com/MLOPS-test/Artifacts/"
    "refs/heads/main/datasets/Knowledge_Base_for_Airline_Info_and_FAQs.pdf"
)
PDF_PATH = "Knowledge_Base_for_Airline_Info_and_FAQs.pdf"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "airline-faq-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

assert PINECONE_API_KEY, "PINECONE_API_KEY missing in environment."


def download_pdf():
    if os.path.exists(PDF_PATH):
        print(f"[skip] {PDF_PATH} already present.")
        return
    print(f"Downloading {PDF_URL} ...")
    urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    print(f"Saved to {PDF_PATH}.")


def main():
    download_pdf()

    print("Loading and chunking PDF ...")
    documents = PyMuPDFLoader(PDF_PATH).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"  pages: {len(documents)}  chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index {PINECONE_INDEX_NAME} ...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    print(f"Uploading {len(chunks)} chunks to {PINECONE_INDEX_NAME} ...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    print("Done.")


if __name__ == "__main__":
    main()
