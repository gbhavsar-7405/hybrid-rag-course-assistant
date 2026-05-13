import os
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
load_dotenv()


# =====================================================
# LOAD CSV DOCUMENTS
# =====================================================


def load_csv_documents(data_path="Datasets"):

    print(f"Loading CSV files from {data_path}...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The directory {data_path} does not exist.")

    documents = []

    for file in os.listdir(data_path):

        if file.endswith(".csv"):

            file_path = os.path.join(data_path, file)

            print(f"\nLoading file: {file}")

            df = pd.read_csv(file_path)

            print(df.columns)
            print(f"Rows found: {len(df)}")

            # =====================================================
            # COURSERA DATASET
            # =====================================================

            if file == "coursera_courses.csv":

                for index, row in df.iterrows():

                    course_name = row.get("course_title", "Unknown Course")

                    difficulty = row.get("course_difficulty", "Not Available")

                    rating = row.get("course_rating", "Not Available")

                    url = row.get("course_url", "Not Available")

                    content = f"""
Course Name: {course_name}
Platform: Coursera
Skills/Topics: {row.get('course_skills', 'Not Available')}
Difficulty Level: {difficulty}
Rating: {rating}
Duration: {row.get('course_time', 'Not Available')}
Certificate Type: {row.get('course_certificate_type', 'Not Available')}
Students Enrolled: {row.get('course_students_enrolled', 'Not Available')}
Reviews Count: {row.get('course_reviews_num', 'Not Available')}
URL: {url}

Summary:
{row.get('course_summary', 'Not Available')}

Description:
{row.get('course_description', 'Not Available')}
"""

                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file,
                            "row": index,
                            "course_name": course_name,
                            "difficulty": difficulty,
                            "rating": rating,
                            "platform": "Coursera",
                            "url": url,
                        },
                    )

                    documents.append(doc)

            # =====================================================
            # UDEMY DATASET
            # =====================================================

            elif file == "udemy_courses.csv":

                for index, row in df.iterrows():

                    course_name = row.get("course_title", "Unknown Course")

                    difficulty = row.get("level", "Not Available")

                    rating = row.get("num_reviews", "Not Available")

                    url = row.get("url", "Not Available")

                    content = f"""
Course Name: {course_name}
Platform: Udemy
Subject: {row.get('subject', 'Not Available')}
Difficulty Level: {difficulty}
Price: {row.get('price', 'Not Available')}
Subscribers: {row.get('num_subscribers', 'Not Available')}
Reviews: {row.get('num_reviews', 'Not Available')}
Lectures: {row.get('num_lectures', 'Not Available')}
Duration: {row.get('content_duration', 'Not Available')}
Paid Course: {row.get('is_paid', 'Not Available')}
URL: {url}
"""

                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file,
                            "row": index,
                            "course_name": course_name,
                            "difficulty": difficulty,
                            "rating": rating,
                            "platform": "Udemy",
                            "url": url,
                        },
                    )

                    documents.append(doc)

            # =====================================================
            # UDACITY DATASET
            # =====================================================

            elif file == "udacity_courses.csv":

                for index, row in df.iterrows():

                    course_name = row.get("Title", "Unknown Course")

                    difficulty = row.get("Level", "Not Available")

                    rating = row.get("Rating", "Not Available")

                    url = row.get("URL", "Not Available")

                    content = f"""
Course Name: {course_name}
Platform: Udacity
Skills Covered: {row.get('Skills Covered', 'Not Available')}
Difficulty Level: {difficulty}
Duration: {row.get('Duration', 'Not Available')}
Rating: {rating}
Review Count: {row.get('Review Count', 'Not Available')}
Prerequisites: {row.get('Prerequisites', 'Not Available')}
Affiliates: {row.get('Affiliates', 'Not Available')}
URL: {url}

Description:
{row.get('Description', 'Not Available')}
"""

                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file,
                            "row": index,
                            "course_name": course_name,
                            "difficulty": difficulty,
                            "rating": rating,
                            "platform": "Udacity",
                            "url": url,
                        },
                    )

                    documents.append(doc)

    if len(documents) == 0:
        raise FileNotFoundError(f"No CSV data found in {data_path}")

    # =====================================================
    # PREVIEW DOCUMENTS
    # =====================================================

    for i, doc in enumerate(documents[:2]):

        print(f"\nDocument {i + 1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content Preview:\n{doc.page_content[:500]}")
        print(f"Metadata: {doc.metadata}")

    print(f"\nTotal documents loaded: {len(documents)}")

    return documents


# =====================================================
# VECTOR STORE CREATION
# =====================================================


def create_vector_store(documents, persist_directory="db/chroma_db"):

    print("\nCreating embeddings and storing in ChromaDB...")

    embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("\n--- Vector Store Created Successfully ---")
    print(f"Vector DB saved at: {persist_directory}")

    return vectorstore


# =====================================================
# MAIN FUNCTION
# =====================================================


def main():

    print("Main Function")

    documents = load_csv_documents()

    print(f"\nTotal documents for embeddings: {len(documents)}")

    create_vector_store(documents)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
