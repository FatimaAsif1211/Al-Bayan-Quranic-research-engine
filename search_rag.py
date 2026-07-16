import sys
from qdrant_client import QdrantClient
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from ollama import Client

# --- CONFIGURATION ---
collection_name = "quran_verses"

print("🔄 INITIALIZING: Connecting to Local Qdrant DB and Loading Embedding Model...")

# 1. Connect to the existing local database
client_db = QdrantClient(path="qdrant_storage")

# 2. Load the exact same embedding model
model_id = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=False)
print("✅ INITIALIZATION: System is ready for semantic search!")

# 3. Connect to local Ollama server explicitly
ollama_client = Client(host='http://localhost:11434')

def search_and_answer(query_text):
    print(f"\n🔍 SEARCHING DATABASE FOR: '{query_text}'...")
    
    # Generate embedding vector for the user query
    inputs = tokenizer(query_text, padding=True, truncation=True, return_tensors="pt")
    
    # Safely handle ONNX outputs
    inputs_onnx = {k: v.numpy() for k, v in inputs.items()}
    outputs = model.model.run(None, inputs_onnx)
    
    # Extract query vector from ONNX output
    query_vector = outputs[0][0, 0, :].tolist()
    
    # Search the top 3 closest matching verses using standard query_points method
    search_results = client_db.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3
    ).points
    
    if not search_results:
        print("❌ NO RESULTS FOUND: Could not find matching context in the database.")
        return

    # Build the context string from search results for the LLM
    context_data = ""
    print("\n--- [TOP RETRIEVED HOLY VERSES] ---")
    for idx, hit in enumerate(search_results, start=1):
        payload = hit.payload
        print(f"\n[{idx}] Verse ID: {payload['verse_id']}")
        print(f"📖 Arabic: {payload['arabic_text']}")
        print(f"📝 English Translation: {payload['english_translation']}")
        print(f"📖 Database Tafseer Check: {payload.get('tafseer', '⚠️ Tafseer Field Missing In DB!')}")
        
        # Append to context for LLM
        context_data += f"Verse {payload['verse_id']}:\nTranslation: {payload['english_translation']}\nTafseer: {payload['tafseer']}\n\n"
    print("------------------------------------\n")
    
    # 4. Generate response using Ollama LLM
    print("🧠 LLM PROCESSING: Generating answer based ONLY on retrieved context... Please wait...")
    
    system_prompt = (
        "You are an advanced Islamic AI Assistant. Answer the user query comprehensively based "
        "ONLY on the provided Quranic verses and Tafseer context. Do not invent any outside information."
    )
    
    user_prompt = f"Context:\n{context_data}\n\nUser Question: {query_text}\n\nAnswer:"
    
    try:
        response = ollama_client.generate(
            model="myllama3",
            system=system_prompt,
            prompt=user_prompt
        )
        print("\n🤖 [AI RESPONSE]:")
        print(response['response'])
    except Exception as e:
        print(f"\n❌ OLLAMA ERROR: Make sure Ollama app is running and 'myllama3' is created. Details: {e}")

if __name__ == "__main__":
    test_query = "What does the Quran say about patience and testing believers?"
    search_and_answer(test_query) 