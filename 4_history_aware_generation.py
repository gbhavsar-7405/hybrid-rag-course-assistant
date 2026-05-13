import os
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.schema import HumanMessage
from langchain.schema import AIMessage
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import (
    ChatNVIDIA,
    NVIDIAEmbeddings,
)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

load_dotenv()

# =====================================================
# LOAD DOCUMENTS FOR BM25
# =====================================================

def load_documents(data_path="Datasets"):
    documents = []
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            file_path = os.path.join(
                data_path,
                file,
            )
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                content = str(row.to_dict())
                documents.append(Document(page_content=content))

    return documents


# =====================================================
# VECTOR DB
# =====================================================

PERSIST_DIRECTORY = "db/chroma_db"
embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

# Load Chroma DB
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
)

# =====================================================
# BM25 RETRIEVER
# =====================================================

print("Loading BM25 documents...")
documents = load_documents()
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 4


# =====================================================
# LLM
# =====================================================

model = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

# =====================================================
# CHAT HISTORY
# =====================================================

chat_history = []

# =====================================================
# ASK QUESTION FUNCTION
# =====================================================


def ask_question(user_question):
    global chat_history

    # =====================================================
    # QUERY REFORMULATION
    # =====================================================
    followup_words = [
        "yes",
        "more",
        "continue",
        "more options",
        "other options",
    ]
    # Reformulate only when needed
    if (
        len(user_question.split()) <= 4
        or user_question.lower().strip() in followup_words
        or len(chat_history) > 2
    ):
        reformulation_prompt = f"""
You are a query reformulation assistant.
Your job is to convert the user's latest question
into a clear standalone search query.
Rules:
- Keep the meaning EXACTLY the same
- Correct spelling mistakes
- NEVER introduce new platforms
- NEVER invent information
- Keep it SHORT and retrieval-friendly
- If the user asks follow-up questions,
  use chat history for context
Chat History:
{chat_history}
User Question:
{user_question}
Standalone Search Query:
"""
        reformulated_query = model.invoke(reformulation_prompt).content.strip()
    else:
        reformulated_query = user_question
    print(f"\nReformulated Query:\n" f"{reformulated_query}")
    # =====================================================
    # DYNAMIC RETRIEVAL DEPTH
    # =====================================================

    k_value = 5

    if any(
        word in user_question.lower()
        for word in [
            "more",
            "10",
            "many",
            "list",
            "options",
        ]
    ):
        k_value = 10

    # =====================================================
    # PLATFORM FILTERING
    # =====================================================

    platform_filter = None
    if "udemy" in user_question.lower():
        platform_filter = "Udemy"

    elif "coursera" in user_question.lower():
        platform_filter = "Coursera"

    elif "udacity" in user_question.lower():
        platform_filter = "Udacity"

    # =====================================================
    # SEARCH SETTINGS
    # =====================================================

    search_kwargs = {
        "k": k_value,
        "fetch_k": 12,
        "lambda_mult": 0.8,
    }

    # =====================================================
    # APPLY METADATA FILTER
    # =====================================================

    if platform_filter:

        search_kwargs["filter"] = {"platform": platform_filter}

    # =====================================================
    # VECTOR RETRIEVER
    # =====================================================

    vector_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

    # =====================================================
    # HYBRID SEARCH
    # =====================================================

    retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_retriever,
        ],
        weights=[0.35, 0.65],
    )

    # =====================================================
    # RETRIEVE DOCUMENTS
    # =====================================================

    relevant_docs = retriever.invoke(reformulated_query)

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    unique_docs = []
    seen_content = set()
    for doc in relevant_docs:
        content = doc.page_content
        if content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(content)
    relevant_docs = unique_docs

    # =====================================================
    # RERANKING
    # =====================================================

    def calculate_score(doc):
        score = 0
        difficulty = str(doc.metadata.get("difficulty", "")).lower()

        rating = str(doc.metadata.get("rating", "0"))
        content = doc.page_content.lower()
        # Beginner boost

        if "beginner" in difficulty:
            score += 3

        # Rating boost
        try:
            rating_value = float(rating)
            if rating_value >= 4.7:
                score += 3
            elif rating_value >= 4.5:
                score += 2
        except:
            pass
        # Keyword match boost

        for word in reformulated_query.lower().split():
            if word in content:
                score += 1
        return score

    relevant_docs = sorted(
        relevant_docs,
        key=calculate_score,
        reverse=True,
    )

    # =====================================================
    # EMPTY RETRIEVAL GUARD
    # =====================================================
    if len(relevant_docs) == 0:
        print("\nNo relevant documents found.")
        return

    # =====================================================
    # CONTEXT CREATION
    # =====================================================

    formatted_contexts = []
    for doc in relevant_docs:
        course_name = doc.metadata.get(
            "course_name",
            "Unknown Course",
        )

        platform = doc.metadata.get(
            "platform",
            "Unknown",
        )

        difficulty = doc.metadata.get(
            "difficulty",
            "Not Available",
        )

        rating = doc.metadata.get(
            "rating",
            "Not Available",
        )

        url = doc.metadata.get(
            "url",
            "Not Available",
        )

        description = doc.page_content[:250]

        formatted_contexts.append(
            f"""
Course Name: {course_name}
Platform: {platform}
Difficulty: {difficulty}
Rating: {rating}
URL: {url}

Description:
{description}
"""
        )

    context = "\n\n".join(formatted_contexts)

    # =====================================================
    # FINAL PROMPT
    # =====================================================

    final_prompt = f"""
    You are an AI Course Recommendation Assistant.

    ONLY answer using the retrieved context.

    If the answer is not present in the context,
    say clearly that you could not find enough
    relevant information.
    - Do NOT recommend unrelated alternatives
    - Do NOT switch platforms unless explicitly asked
    - If relevant courses are unavailable,
    clearly say so
    - Do NOT hallucinate.
    - Do NOT invent courses.
    - Do NOT invent URLs.
    - Do NOT recommend loosely related courses.

    - If the user asks follow-up queries like:
    "more", "continue", "other options",
    continue the previous recommendation flow

    Format responses like:

    1. Course Name
    ⭐ Rating
    🎯 Difficulty
    🔗 URL
    📝 Short Description

    Retrieved Context:
    {context}
    Conversation History:
    {chat_history}
    User Question:
    {user_question}

    Answer:
    """

    result = model.invoke(final_prompt)

    print("\n" + "=" * 60)

    print("🤖 AI Assistant:\n")

    print(result.content)

    print("\n" + "=" * 60)

    # =====================================================
    # SAVE CHAT HISTORY
    # =====================================================

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=result.content))


# =====================================================
# CHAT LOOP
# =====================================================

print("History-Aware RAG Chat Started")
print("Type 'quit' to exit.\n")
while True:
    user_question = input("\n🧑 You: ")
    if user_question.lower() == "quit":
        break
    ask_question(user_question)
