import os
from langchain.docstore.document import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from fuzzywuzzy import process
from together import Together

# Initialize Together client
client = Together()

# To convert the text to numbers
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the vector store
vectorstore = Chroma(

    collection_name="resumes",
    embedding_function=embedding_model,
    persist_directory="resumes_chroma",

)

def get_all_candidate_names():

    collection = vectorstore.get() # Each info you store is a document and 'collection' is all the list of docs
    return list({doc.get('candidate', '').lower() for doc in collection['metadatas']}) # Each doc has a metadata field

def split_text(documents: list[Document]):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        add_start_index=True,   # Save position of the first token in the chunk w.r.t original text not necessary
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks (chunk_size={500}, overlap={75}).")
    
    if chunks:
        print("\nSample Chunk:\n\n", chunks[3].page_content[:500]) 

    return chunks

def upload_resume():

    print("📂 Drag and drop the PDF file here, then press Enter:")

    raw_path = input().strip().strip('"').strip("'")

    # Remove leading '& ' if present
    if raw_path.startswith("& "):
        raw_path = raw_path[2:]

    file_path = raw_path.strip('"').strip("'")

    print(f"\n📄 Selected file: {file_path}")

    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()   # This variable is an array of document objects ( objects = no. of pages) and each doc contains 'text' and 'metadata'

    chunks = split_text(documents)
    candidate_name = os.path.splitext(os.path.basename(file_path))[0].lower()   # save document as lowercase name

    document_objects = [
        Document(page_content=chunk.page_content, metadata={"candidate": candidate_name})
        for chunk in chunks
    ]

    vectorstore.add_documents(document_objects)
    print(f"\n✅Resume for '{candidate_name}' stored successfully.")

def query_resume_with_rag(query_text: str):

    candidate_list = get_all_candidate_names()
    candidate, score = process.extractOne(query_text.lower(), candidate_list)   # identify name from names (candidate_list) in the vector db

    if score < 60:
        print(" Could not identify a relevant candidate from your query.")
        return

    print(f" Matched candidate: {candidate} (Confidence: {score}%)")

    results = vectorstore.similarity_search(query_text, k=3, filter={"candidate": candidate})

    # Identifies name from query and then searches for chunks within only the document with that name

    if not results:
        print("No relevant chunks found.")
        return

    context = "\n\n".join([doc.page_content for doc in results])
    print("le context",context)

    messages = [
        {"role": "system", "content": "You are a helpful assistant answering questions about a resume. Answer ONLY in a short direct form. No full sentences, no repetition of the question, no explanations."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
    ]

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        max_tokens=512,
        temperature=0.6
    )

    print(f"\nLLM Answer:\n{response.choices[0].message.content}")

def main():

    while True:

        print("\n===== Resume Chatbot CLI =====")
        print("1. Upload a resume")
        print("2. Ask a question")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            upload_resume()
        elif choice == "2":
            query = input("Ask your question: ").strip()
            query_resume_with_rag(query)
        elif choice == "3":
            break

if __name__ == "__main__":
    main()
