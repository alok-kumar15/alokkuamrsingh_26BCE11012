import os
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Extract text from PDF
reader = PdfReader("document.pdf")
text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

# Split into chunks
chunk_size = 500
chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i + chunk_size])

print("Number of chunks:", len(chunks))

# Generate embeddings and store in FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Embeddings stored successfully.")

# Query processing
query = input("\nAsk a question: ")

query_embedding = model.encode([query])

k = 3
distances, indices = index.search(query_embedding, k)

context = ""

for i in indices[0]:
    context += chunks[i] + "\n\n"

# Initialize Gemini Client (automatically picks up GEMINI_API_KEY from environment)
client = genai.Client()

prompt = f"""
Answer the question using only the information provided in the context.

If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

# Call Gemini model
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
)

print("\nAnswer: ")
print(response.text)