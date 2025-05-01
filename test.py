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
        chunk_overlap=0,
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

    print(f"Matched candidate: {candidate} (Confidence: {score}%)")

    results = vectorstore.similarity_search(query_text, k=3, filter={"candidate": candidate})

    # Identifies name from query and then searches for chunks within only the document with that name

    if not results:
        print("No relevant chunks found.")
        return

    context = "\n\n".join([doc.page_content for doc in results])

    messages = [
    {
        "role": "system",
        "content": (
            "You are an AI assistant helping with resume data extraction."
            "Answer briefly, directly, and without any explanation or commentary. "
            "If the answer is a eg. number or a date or a bunch of certificates, return only that. "
            "Do NOT restate the question. Do NOT explain the answer. Return just the answer."
            "The current year is 2025"
        )
    },
    {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query_text}\nAnswer:"
    }
    ]


    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        messages=messages,
        max_tokens=512,
        temperature=0.6
    )

    response_text = response.choices[0].message.content

    start_tag = "<think>"
    end_tag = "</think>"

    start_index = response_text.find(start_tag)
    end_index = response_text.find(end_tag)

    if start_index != -1 and end_index != -1:
        think_content = response_text[start_index + len(start_tag):end_index].strip()
        final_answer = response_text[end_index + len(end_tag):].strip()
    else:
        think_content = None
        final_answer = response_text.strip()

    ''' Output both
    print("\nReasoning:\n")
    print(think_content if think_content else "[No <think> section found]")'''

    print("\nLLM Answer:\n")
    print(final_answer)


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
