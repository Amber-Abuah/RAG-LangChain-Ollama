from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import HumanMessage

pdf_path = "Documents/CatWiki.pdf"

loader = UnstructuredPDFLoader(pdf_path)
data = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
chunks = splitter.split_documents(data)
print("Loaded and split documents into chunks.")

print("Creating Chroma database...")
vector_database = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="local-rag"
)

print("Database creation finished.")

model_name = "llama3"
llm = ChatOllama(model=model_name)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_database.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chat_history = []
user_query = ""

while user_query != "exit":
    user_query = input("\nPlease ask the LLM a question about the document: ")
    answer = chain.invoke({"input": user_query, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=user_query), answer])
    print(answer)