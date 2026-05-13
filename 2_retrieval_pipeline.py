from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()

# Connect to ChromaDB
persistent_directory = "db/chroma_db"

embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
)

# User query
while True:

    query = input("\nEnter your query: ")

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

print(f"\nUser Query: {query}")
print("\n---- Relevant Results ----")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\nResult {i}")
    print(f"Source: {doc.metadata['source']}")
    print(f"Row: {doc.metadata['row']}")
    print("\nContent:")
    print(doc.page_content[:1500])
    print("-" * 50)


# Synthetic Questions:

# Beginner courses
"Best beginner level Python courses"

# AI / ML
"Top machine learning courses for beginners"

# Web Development
"Courses that teach React and APIs"

# Cybersecurity
"Best cybersecurity certification programs"

# Data Science
"Courses covering pandas and data analysis"

# Cloud Computing
"AWS or cloud computing related courses"

# Java
"Intermediate Java programming courses"

# Free courses
"Free web development courses"

# Advanced courses
"Advanced deep learning specializations"

# Short duration
"Courses under 10 hours for Python"

# Career focused
"Full stack developer career courses"

# University style
"Professional certificate programs from Coursera"

# Practical learning
"Hands-on projects in AI courses"

# Beginner friendly
"Easy courses for absolute coding beginners"

# Certification based
"Courses with completion certificates"
