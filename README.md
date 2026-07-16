# Al Bayan: A Local Retrieval Augmented Research Engine for the Tafsir of Ibn Kathir

## Abstract

Al Bayan is an offline, locally hosted research application for the study of Qur'anic exegesis. It provides semantic retrieval, direct verse access, and exact term matching across the Tafsir of Ibn Kathir, supported by a vector database and a locally hosted language model for scholarly synthesis. The system is designed for reproducibility and privacy, as no data leaves the host machine once the environment is configured.
## Table of Contents

1. Overview
2. Methodology
3. System Architecture
4. Interface and Functionality
5. Technology Stack
6. Requirements
7. Installation and Setup
8. Limitations and Scope
9. Performance Notes

## 1. Overview

The application allows a researcher to pose a conceptual question in natural language and receive a synthesized scholarly answer supported by cross referenced verses and their corresponding commentary. It also supports precise lookup by chapter and verse coordinates, literal keyword matching with inline highlighting, and administrative ingestion of new source records into the vector store. All computation, including embedding generation and language model inference, is performed locally.

## 2. Methodology

This project follows a Retrieval Augmented Generation approach. Rather than relying on a language model to answer questions from memory, the system first retrieves relevant passages from a verified source text, the Tafsir of Ibn Kathir, and then generates an answer grounded in those retrieved passages. This design was chosen specifically to reduce the risk of unsupported or fabricated claims, since every generated statement can be traced back to a cited source record.

The methodology consists of the following stages.

**Embedding.** Each text segment is converted into a dense vector representation using the BGE M3 embedding model, which captures semantic meaning rather than exact word matching.

**Indexing.** The resulting vectors are stored in a Qdrant vector database, which supports efficient similarity search across the full corpus.

**Retrieval.** When a researcher submits a query, the query itself is embedded using the same model, and the system retrieves a pool of candidate passages ranked by vector similarity.

**Re ranking.** A cross encoder model, MiniLM, re scores the retrieved candidates against the original query to improve the precision of the final ranked list before it is passed to the language model.

**Synthesis.** The top ranked passages are provided as context to a locally hosted language model through Ollama, which produces a scholarly summary with inline citations to the supporting verses.

## 3. System Architecture

The application follows a retrieval augmented generation design. User input is embedded using a dense representation model, compared against a locally stored vector index, and the retrieved passages are passed to a local language model that produces a grounded, citation supported summary. Four operating modes are exposed through the interface: Semantic Search and Synthesis, Direct Verse Lookup, Keyword Scan, and Data Ingestion.

## 4. Interface and Functionality

### 4.1 Semantic Search and Synthesis

A researcher submits a conceptual query, such as "kindness," and the system retrieves the most relevant passages from the indexed tafsir, ranks them by similarity score, and generates a scholarly summary that cites each supporting record by surah, ayah, and translation.

![Semantic search interface showing the query and the beginning of the scholarly synthesis](docs/screenshots/semantic-search.png)

The synthesis continues with each cited verse presented alongside its original Arabic text and an English rendering, allowing the researcher to verify every claim against its source.

![Semantic search results continued, showing supporting Arabic text and translated evidence records](docs/screenshots/semantic-search_extended.png)

The final section of the synthesis presents a conclusion, a numbered reference list, and reports the retrieval latency, the number of candidates scanned, and the number of verified matches. A downloadable plain text summary is also provided.

![Concluding synthesis, reference list, and performance metrics for the semantic search query](docs/screenshots/semantic-search_ext2.png)

### 4.2 Direct Verse Lookup

A researcher may enter a standard chapter to verse coordinate, for example 7:199, and the system returns the exact matching Arabic text together with its English translation.

![Direct verse lookup showing Al A'raaf, Ayah 199, with Arabic text and translation](docs/screenshots/direct-verse_lookup.png)

### 4.3 Keyword Scan

The keyword scan mode performs an exact term search across the indexed corpus and highlights every occurrence of the search term within the returned passages, along with a count of matching records.

![Keyword scan input field showing the search term water and eighteen matching records](docs/screenshots/keyword-scan.png)

![Additional matches from the keyword scan, with the search term highlighted in context](docs/screenshots/keyword-scan_cont.png)

![Further keyword scan results spanning multiple surahs](docs/screenshots/keyword-scan_cont2.png)

### 4.4 Live Database Inspector

This mode allows a researcher to browse and paginate through the raw records stored in the vector database, which is useful for verifying that ingestion completed correctly and that record fields are populated as expected.

![Live Database Inspector showing paginated records retrieved from the vector database](docs/screenshots/database-inspect.png)

![Expanded database record showing the full English and Urdu tafsir fields](docs/screenshots/database-inspect-extended.png)

### 4.5 Data Ingestion

New records can be added to the index by uploading a structured CSV file containing the required fields. The interface previews the uploaded data before it is appended and indexed.

![Data ingestion pipeline with instructions for uploading and indexing a CSV file](docs/screenshots/data-ingestion.png)

## 5. Technology Stack

| Component | Purpose |
|---|---|
| Streamlit | User interface |
| Qdrant | Local vector database |
| BGE M3 | Embedding model, served through ONNX Runtime |
| Cross encoder MiniLM | Re ranking of retrieved candidates |
| Ollama | Local language model for scholarly synthesis |

## 6. Requirements

### 6.1 Software

Python 3.11 has been used for development and testing. Ollama must be installed and running locally with the required model pulled, for example by running `ollama pull llama3`. The application expects a local model named `myllama3` by default; this can be changed through the `OLLAMA_MODEL_NAME` setting in `app.py`. All Python dependencies are listed in `requirements.txt`.

### 6.2 Hardware

A minimum of eight gigabytes of system memory is required, with sixteen gigabytes recommended. Approximately five to six gigabytes of free disk space are needed for model weights and the ONNX export, in addition to storage required by the Qdrant index.

A GPU is optional. By default, the application runs entirely on the CPU using ONNX Runtime. Installing `onnxruntime-gpu` in place of `onnxruntime` enables accelerated embedding generation on a CUDA capable NVIDIA GPU with matching CUDA and cuDNN versions. The local language model also benefits from GPU acceleration but will function on CPU alone.

## 7. Installation and Setup

Clone the repository and enter its directory.

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

Install the required Python packages.

```bash
pip install -r requirements.txt
```

Start Ollama with the local model in a separate terminal.

```bash
ollama run myllama3
```

Launch the application.

```bash
streamlit run app.py
```

On first launch, the application exports and caches the BGE M3 model to ONNX format in the directory `models/bge-m3-onnx`. This directory is excluded from version control through the `.gitignore` file.

To enable GPU acceleration, replace `optimum[onnxruntime]` with `onnxruntime-gpu`.

Note on data: the source dataset, including the tafsir CSV, the per surah JSON files, and the original PDF material, is not included in this repository. To populate the vector database, prepare a structured CSV containing the required fields and use the Data Ingestion mode described in Section 4.5 to build the index locally.

## 8. Limitations and Scope

This system is intended as a research and study aid, not as a source of religious ruling or authoritative interpretation. The language model synthesis step, while grounded in retrieved source passages, can still misrepresent nuance present in the original tafsir. Researchers are advised to treat every generated summary as a starting point and to verify each cited claim against the displayed source text before using it in any formal work. The quality of retrieval also depends on the completeness and accuracy of the ingested dataset; incomplete or poorly segmented source data will reduce the relevance of retrieved passages.

## 9. Performance Notes

This project was developed and evaluated on a GPU enabled workstation. Embedding generation was accelerated with `onnxruntime-gpu` using CUDA, and the local language model achieved faster response times under GPU execution than under CPU execution. The application remains fully functional on CPU alone, though summarization and semantic search complete more slowly in that configuration.
 

 