import glob
import os
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

# 2. Document Loading
doc_folder = os.path.join(os.path.dirname(__file__), "document_loader")
pdf_path = os.path.join(doc_folder, "STATS.pdf")

if not os.path.exists(pdf_path):
    pdfs = glob.glob(os.path.join(doc_folder, "*.pdf"))
    if pdfs:
        pdf_path = pdfs[0]
    else:
        raise FileNotFoundError(f"No PDF found in: {doc_folder}")

print(f"Loading document from: {pdf_path}")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} document pages.")

# 3. Document Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} text chunks.")

# 4. Embeddings & Vector Store
print("Creating embeddings and indexing into vector store...")
embeddings = FastEmbedEmbeddings()
vectorstore = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
)
print("Indexing complete!")

# 5. Initialize NVIDIA LLM
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    nvidia_api_key=api_key,
    temperature=0.2,
    max_completion_tokens=2500,
)

# 6. Retriever & RAG Prompt
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert tutor in Statistical Techniques. "
        "Answer the question accurately based ONLY on the provided context. "
        "If the answer cannot be found in the context, say that you don't know.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}"),
])

rag_chain = prompt | llm | StrOutputParser()


def query_rag(question: str):
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    retrieved_docs = retriever.invoke(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant chunk(s):")
    for i, doc in enumerate(retrieved_docs, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        preview = doc.page_content.strip().replace("\n", " ")[:150]
        print(f"  [{i}] (Page {page_num}): {preview}...")

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    print("\nGenerating answer...")
    response = rag_chain.invoke({"context": context, "question": question})
    print("\nAnswer:")
    print(response)
    print("=" * 60 + "\n")
    return response


def chat_loop():
    print("\nPDF RAG ready! Ask questions (type 'exit' or 'quit' to stop).\n")
    while True:
        try:
            question = input("Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Session ended.")
            break

        query_rag(question)


if __name__ == "__main__":
    chat_loop()
    