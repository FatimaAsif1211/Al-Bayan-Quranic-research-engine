import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

# --- CONFIGURATION MATCHING YOUR EXACT FILE ---
CSV_FILE_PATH = "final_dataset_surah_tafseer_verified1.csv"  # Updated file name
COL_TEXT_TO_EMBED = "en"          # We will search using the English translation column
COL_ARABIC = "arabic"             # Arabic column name
COL_TAFSEER = "tafseer"           # Tafseer column name
COL_VERSE_ID = "verse_id"         # Unique Verse ID column (e.g., 1:1)

print("🔄 INITIALIZING: Loading BGE-M3 ONNX Model and Qdrant Local Database...")

# 1. Initialize Qdrant Client (Persistent Local Storage)
client = QdrantClient(path="qdrant_storage") 

# Create or re-create collection
collection_name = "quran_verses"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)
print("✅ QDRANT: Local storage collection 'quran_verses' created successfully.")

# 2. Initialize Embedding Model (BGE-M3 ONNX Runtime)
model_id = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
print("✅ MODEL: BGE-M3 ONNX model loaded successfully.")

# 3. Load and Process CSV Data
if not os.path.exists(CSV_FILE_PATH):
    print(f"❌ ERROR: File '{CSV_FILE_PATH}' not found! Please place your CSV file in the project folder.")
    exit()

print(f"🔄 CSV: Loading data from '{CSV_FILE_PATH}'...")
df = pd.read_csv(CSV_FILE_PATH)
print(f"✅ CSV: Loaded successfully. Found {len(df)} rows.")

# 4. Generate Embeddings and Upsert to Database
print("🔄 PROCESSING: Generating embeddings and indexing data into Qdrant... Please wait...")
points = []

for idx, row in df.iterrows():
    # Safely extract text content and metadata from the row
    text_content = str(row[COL_TEXT_TO_EMBED])
    arabic_text = str(row[COL_ARABIC])
    tafseer_text = str(row[COL_TAFSEER])
    verse_id_str = str(row[COL_VERSE_ID])
    
    # Generate vector tokens using BGE-M3
    inputs = tokenizer(text_content, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    
    # Extract 1024-dimension embedding vector
    embedding_vector = outputs.last_hidden_state[:, 0, :].detach().numpy()[0].tolist()
    
    # Prepare payload with rich metadata
    point = PointStruct(
        id=idx,  # Unique integer ID for Qdrant points
        vector=embedding_vector,
        payload={
            "english_translation": text_content,
            "arabic_text": arabic_text,
            "tafseer": tafseer_text,
            "verse_id": verse_id_str
        }
    )
    points.append(point)

# Upload all vectors into Qdrant database
client.upsert(collection_name=collection_name, points=points)
print(f"🚀 SUCCESS: Database population complete! {len(points)} verses indexed successfully.")