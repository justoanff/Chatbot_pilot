import os
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.2",
)
os.environ['USER_AGENT'] = 'myagent'

async def load_and_split_data(file_paths, chunk_size=500, chunk_overlap=100):
    data = []
    successfully_loaded_files = []
    failed_files = []

    for file_path in file_paths:
        try:
            loader = PyPDFLoader(file_path)
            async for page in loader.alazy_load():
                data.append(page)
            successfully_loaded_files.append(file_path)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            failed_files.append(file_path)

    if data:
        print("Documents loaded successfully.")
    else:
        print("Failed to load documents.")

    print(f"Successfully loaded files: {successfully_loaded_files}")
    print(f"Failed files: {failed_files}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)

def create_vectorstore(documents, model_name="nomic-embed-text"):
    local_embeddings = OllamaEmbeddings(model=model_name)
    return Chroma.from_documents(documents=documents, embedding=local_embeddings)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_chain():
    RAG_TEMPLATE = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    <context>
    {context}
    </context>

    Answer the following question:

    {question}"""

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    model = ChatOllama(model="llama3.2")

    return (
            RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
            | rag_prompt
            | model
            | StrOutputParser()
    )

async def chatbot(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist.")
        return

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not file_paths:
        print(f"No PDF files found in the folder '{folder_path}'.")
        return

    all_splits = await load_and_split_data(file_paths)
    vectorstore = create_vectorstore(all_splits)
    chain = create_chain()

    print("Chatbot is ready. Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.lower() == 'exit':
            break
        docs = vectorstore.similarity_search(question)
        response = chain.invoke({"context": docs, "question": question})
        print("Bot:", response)

# Run the chatbot
asyncio.run(chatbot("/Users/justoanff/PycharmProjects/Langchain_RAG/data"))