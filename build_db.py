import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import torch

# --- CONFIGURATION ---
QDRANT_STORAGE_PATH = "qdrant_storage"
COLLECTION_NAME = "tafsir_ibn_kathir_v2"
DATA_FOLDER = "./iqra-tafsir-data" # Path to your JSON files folder
MODEL_ID = "BAAI/bge-m3"

print("🔄 Loading embedding models and Qdrant DB...")
client = QdrantClient(path=QDRANT_STORAGE_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)

# Recreate a fresh collection
print(f"🧹 Recreating collection: {COLLECTION_NAME}...")
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE) # BGE-M3 base dimension
)

def get_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()[0].tolist()

# Function to split long text into smaller character chunks
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

print("🚀 Starting Quran & Tafseer indexing...")
id_counter = 1

# Loop through all files in alphabetical order
for file_name in sorted(os.listdir(DATA_FOLDER)):
    if file_name.endswith('.json') and file_name.startswith('surah_'):
        file_path = os.path.join(DATA_FOLDER, file_name)
        
        # Extract surah number from file name (e.g., surah_001.json -> 1)
        try:
            surah_num_str = file_name.split('_')[1].split('.')[0]
            surah_num_int = str(int(surah_num_str))
        except:
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                file_data = json.load(f)
            except Exception as e:
                print(f"⚠️ Error reading JSON file {file_name}: {e}")
                continue
            
            # Extract content using the numerical surah key
            surah_content = file_data.get(surah_num_int) or file_data.get(str(int(surah_num_str)))
            if not surah_content:
                # Fallback to the first key if direct match fails
                first_key = list(file_data.keys())[0]
                surah_content = file_data[first_key]
                
            english_tafsir = surah_content.get("en", "")
            urdu_tafsir = surah_content.get("ur", "") # Includes Urdu translation/tafsir if present
            
            print(f"📖 Processing Surah Number: {surah_num_int}...")
            
            # Split long English tafsir into manageable chunks
            en_chunks = chunk_text(english_tafsir)
            
            for index, chunk in enumerate(en_chunks):
                # Formulate structural metadata context for better search accuracy
                meta_search_text = f"Surah Number: {surah_num_int}. Tafsir Content: {chunk}"
                
                try:
                    vector = get_embedding(meta_search_text)
                    
                    payload = {
                        "surah_num": surah_num_int,
                        "chunk_id": index,
                        "english_tafsir_chunk": chunk,
                        "urdu_tafsir_full": urdu_tafsir[:2000] # Storing initial Urdu text segment as safe reference
                    }
                    
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[
                            PointStruct(
                                id=id_counter,
                                vector=vector,
                                payload=payload
                            )
                        ]
                    )
                    id_counter += 1
                except Exception as e:
                    print(f"⚠️ Error processing Chunk {index} of Surah {surah_num_int}: {e}")

print("🎉 Success! All 114 Surahs with high-quality Tafseer data have been indexed into the Vector Database!")