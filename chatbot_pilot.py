import os
from langchain_community.document_loaders import WebBaseLoader
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
# Set the USER_AGENT environment variable
os.environ['USER_AGENT'] = 'myagent'


# Function to load and split data
def load_and_split_data(url, chunk_size=500, chunk_overlap=100):
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)


# Function to create embeddings and vector store
def create_vectorstore(documents, model_name="nomic-embed-text"):
    local_embeddings = OllamaEmbeddings(model=model_name)
    return Chroma.from_documents(documents=documents, embedding=local_embeddings)


# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Function to create the chain
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


# Chatbot interaction loop
def chatbot(url):
    all_splits = load_and_split_data(url)
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
chatbot("https://lilianweng.github.io/posts/2023-06-23-agent/")
