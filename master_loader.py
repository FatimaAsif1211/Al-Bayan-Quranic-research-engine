import os
import re
import fitz  # PyMuPDF
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

# --- CONFIGURATION ---
CSV_PATH = "Quran_Full_Dataset.csv"  #  Quran CSV file
PDFS_FOLDER = "./pdfs"
QDRANT_STORAGE_PATH = "qdrant_storage"
COLLECTION_NAME = "tafsir_ibn_kathir_v2"
EMBEDDING_DIM = 1024  # BGE-M3 model size

print("🔄 INITIALIZING: Loading BGE-M3 ONNX Model and Connecting to Local Qdrant...")

# 1. Connect to Qdrant
client = QdrantClient(path=QDRANT_STORAGE_PATH)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
)
print(f"✅ QDRANT: Collection '{COLLECTION_NAME}' successfully created.")

# 2. Load BGE-M3 ONNX Model
model_id = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
print("✅ MODEL: Real BGE-M3 ONNX model loaded successfully.")


def deep_clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\bbbb+\b', '', text)
    text = re.sub(r'[O¼ِ½_¾ُْﻞُّﺗَﻨْ]', '', text)
    text = re.sub(r'[ﲑﺜﮐﻦﺑاﲑﺴﻔ]', '', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def extract_tafseer_map_from_pdf(pdf_path):
    """PDF ko scan karke 'Ayah Number' -> 'Tafseer Text' ka ek dictionary map banata hai"""
    if not os.path.exists(pdf_path):
        return {}
        
    doc = fitz.open(pdf_path)
    all_raw_text = []
    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            cleaned = deep_clean_text(b[4])
            if cleaned:
                all_raw_text.append(cleaned)
                
    full_text = "\n".join(all_raw_text)
    
    # Regex split: Find ayat numbers and split (e.g., '1.', '2.')
    pattern = r'(?:\n|^)(\d+)\.\s+'
    splits = re.split(pattern, full_text)
    
    tafseer_map = {}
    if len(splits) > 1:
        for i in range(1, len(splits), 2):
            try:
                verse_num = int(splits[i])
                content = splits[i+1].strip()
                
                #Extract Tafseer transition point 
                tafseer_indicators = ["Allah says:", "Allah tells us", "The Command", "Allah commands", "Allah mentions", "This is why"]
                tafseer_text = ""
                
                for indicator in tafseer_indicators:
                    if indicator in content:
                        parts = content.split(indicator, 1)
                        tafseer_text = indicator + " " + parts[1].strip()
                        break
                
                if not tafseer_text:
                    tafseer_text = content  # If no indicator found, use full content as tafseer
                    
                tafseer_map[verse_num] = tafseer_text
            except Exception:
                continue
                
    return tafseer_map


def get_real_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()[0].tolist()


def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: CSV File '{CSV_PATH}' nahi mili!")
        return
        
    # Read CSV
    df = pd.read_csv(CSV_PATH)
    print(f"📂 CSV Loaded: Total {len(df)} verses found.")
    
    # create a mapping of Surah numbers to their corresponding PDF files for quick access
    pdf_files = {int(f.split()[0]): f for f in os.listdir(PDFS_FOLDER) if f.endswith(".pdf") and f.split()[0].isdigit()}

    all_points = []
    current_pdf_num = -1
    current_tafseer_map = {}

    print("\n🚀 STARTING HYBRID PIPELINE...")

    for idx, row in df.iterrows():
        surah_id = int(row['Surah_ID'])
        ayah_id = int(row['Ayah_ID'])
        ayah_ar = str(row['Ayah_Text_Ar'])
        ayah_en = str(row['Ayah_Text_En'])
        surah_name_en = str(row['Surah_Name_En'])
        
        # Check if we need to load a new PDF for the current Surah
        if surah_id != current_pdf_num:
            current_pdf_num = surah_id
            if surah_id in pdf_files:
                pdf_path = os.path.join(PDFS_FOLDER, pdf_files[surah_id])
                print(f"📄 Processing PDF for Surah {surah_id:03d} ({surah_name_en})...")
                current_tafseer_map = extract_tafseer_map_from_pdf(pdf_path)
            else:
                print(f"⚠️ Warning: PDF for Surah {surah_id} not found. Defaulting to empty tafseer.")
                current_tafseer_map = {}

        # Tafseer text retrieval with fallback if not found in the map
        tafseer_text = current_tafseer_map.get(ayah_id, f"Detailed Tafseer for Surah {surah_name_en}, Ayah {ayah_id} is being processed.")

        # Generate embedding for the English translation of the Ayah
        embedding_vector = get_real_embedding(ayah_en)

        # Vector DB point structure
        point = PointStruct(
            id=idx,  # Unique ID for Qdrant (0 to 6235)
            vector=embedding_vector,
            payload={
                "verse_id": f"{surah_id}:{ayah_id}",
                "surah_name": surah_name_en,
                "arabic_text": ayah_ar,
                "english_translation": ayah_en,
                "tafseer": tafseer_text
            }
        )
        all_points.append(point)

        # Batch upload to Qdrant every 100 points to optimize performance
        if len(all_points) >= 100:
            client.upsert(collection_name=COLLECTION_NAME, points=all_points)
            all_points = []
            print(f"🔹 Uploaded: {idx+1}/{len(df)} verses indexed.")

    # Uploading Remaining points 
    if all_points:
        client.upsert(collection_name=COLLECTION_NAME, points=all_points)
        print(f"🔹 Uploaded remaining points. Final Count: {len(df)}")

    print(f"\n🎉 ALHAMDULILLAH: All {len(df)} verses successfully loaded with exact Tafseer mappings in Qdrant!")


if __name__ == "__main__":
    main()