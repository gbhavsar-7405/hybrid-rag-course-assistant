from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Connect to vector DB
persistent_directory = "db/chroma_db"

embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
)

# Query
while True:

    query = input("\n🧑 You: ")

    if query.lower() == "quit":
        print("Goodbye!")
        break

# Retriever
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7,
    },
)

relevant_docs = retriever.invoke(query)

# Combine retrieved context
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# LLM
model = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

# Prompt
system_prompt = """
You are a helpful AI course recommendation assistant.
Only answer using the retrieved context.
If the answer is not present in the context,say:
'I could not find enough relevant information in the database.'
Give concise and structured answers.
"""

human_prompt = f"""
User Query:
{query}

Retrieved Context:
{context}
"""

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=human_prompt),
]

# Generate answer
result = model.invoke(messages)

print("\n--- GENERATED RESPONSE ---\n")
print(result.content)
