# UrbanPulse AI — Multimodal RAG-Based Urban Traffic Intelligence

An AI-powered urban traffic intelligence platform combining multimodal data processing, semantic retrieval, geospatial analysis, traffic analytics, and agentic question answering for Hyderabad.

## 🚀 Live Application

**Live Demo:** [UrbanPulseAI](https://multimodal-rag--urbanpulseai-78jgabwgjk4xil3jyjiygm.streamlit.app/)
**GitHub Repository:** https://github.com/PurpleCoder-23/Multimodal-RAG--UrbanPulseAI

---

## 📌 Overview

UrbanPulse AI is an end-to-end AI application designed to bring together different types of urban traffic information in a single platform.

Traffic-related information is often distributed across images, advisories, structured datasets, weather records, and road network maps. UrbanPulse AI integrates these sources through multimodal data processing, Retrieval-Augmented Generation (RAG), semantic retrieval, and an agentic architecture.

Users can explore urban traffic information through interactive dashboards and ask natural-language questions using the AI assistant.

The project currently focuses on a selected study area in Hyderabad.

## ✨ Key Features

### AI Assistant
- Natural-language question answering using a LangGraph-based agent.
- Retrieval-supported responses based on available project data.
- Direct agent integration within the Streamlit application.

### Multimodal Document Processing
- Traffic advisory image processing.
- OCR and vision-language-based information extraction.
- Preparation of extracted information for retrieval.

### Semantic and Hybrid Retrieval
- Semantic search using Sentence Transformers.
- Hybrid retrieval combining semantic and structured information.
- Context retrieval for traffic-related queries.

### Geospatial Intelligence
- Road network exploration using GeoJSON data.
- Interactive road visualization using Folium.
- Location-based road network analysis.

### Traffic Analytics
- Interactive traffic analytics dashboards.
- Congestion distribution analysis.
- Vehicle flow and speed visualization.
- Plotly-based interactive charts.

### Advisory Intelligence
- Exploration of extracted traffic advisories.
- Junction and diversion information.
- Advisory-related information retrieval.

### Weather Intelligence
- Visualization of stored Hyderabad weather information.
- Exploration of available weather measurements.

---

## 🏗️ System Architecture

The current application follows this workflow:

1. **Data ingestion:** Load traffic records, advisory documents, images, road network data, and weather information.
2. **Multimodal processing:** Extract information from supported documents and images.
3. **Data preparation:** Process and organize records for retrieval and analysis.
4. **Embedding generation:** Generate semantic embeddings using Sentence Transformers.
5. **Retrieval:** Retrieve relevant information through semantic, hybrid, and structured retrieval components.
6. **Agent orchestration:** LangGraph coordinates the question-answering workflow.
7. **Response generation:** Groq-powered language model generates responses based on available context.
8. **Visualization:** Streamlit presents the assistant and interactive analytics dashboards.

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Agentic Orchestration | LangGraph |
| LLM | Groq API |
| Embeddings | Sentence Transformers |
| Retrieval | Semantic, Hybrid, Structured Retrieval |
| Data Processing | Pandas, NumPy |
| Geospatial Processing | Shapely, PyProj |
| Maps | Folium, Streamlit-Folium |
| Visualization | Plotly |
| Image Processing | Pillow, RapidOCR |
| Backend API | FastAPI |
| Dependency Management | uv |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |
| Containerization | Docker |

---

## 📂 Project Structure

```text
UrbanPulseAI/
│
├── data/
│   ├── Advisories/
│   ├── Maps/
│   ├── Weather/
│   ├── processed/
│   └── traffic/
│
├── scripts/
│   ├── analyze_advisory_image.py
│   ├── answer_generator.py
│   ├── api.py
│   ├── build_multimodal_records.py
│   ├── chunk_documents.py
│   ├── clean_roads.py
│   ├── embed_multimodal_records.py
│   ├── generate_traffic.py
│   ├── geospatial_retriever.py
│   ├── hybrid_retriever.py
│   ├── ingest_data.py
│   ├── ingest_images.py
│   ├── intent_router.py
│   ├── retrieve_chunks.py
│   ├── retrieve_data.py
│   ├── retrieve_multimodal.py
│   ├── structured_retriever.py
│   └── urbanpulse_agent.py
│
├── tests/
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env.example
├── .gitignore
└── README.md
```

---

## 💻 Local Installation and Setup

### Prerequisites

- Python 3.12
- Git
- uv
- Groq API key

### 1. Clone the repository

```bash
git clone https://github.com/PurpleCoder-23/Multimodal-RAG--UrbanPulseAI.git
```

### 2. Navigate to the project directory

```bash
cd Multimodal-RAG--UrbanPulseAI
```

### 3. Install dependencies

Install uv if required:

```bash
pip install uv
```

Install project dependencies:

```bash
uv sync
```

### 4. Configure environment variables

Create a `.env` file in the project root.

Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own API key.

**Security:** Never commit your `.env` file or expose API keys in GitHub repositories.

### 5. Run the Streamlit application

```bash
uv run streamlit run streamlit_app.py
```

Open the application:

```text
http://localhost:8501
```

The Streamlit application initializes the LangGraph agent directly. A separate FastAPI server is not required for this configuration.

### 6. Run the optional FastAPI backend

The project also contains a FastAPI backend for local development and future backend-based deployment.

Start the backend:

```bash
uv run uvicorn scripts.api:app --reload
```

Backend URL:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

Health endpoint:

```text
http://localhost:8000/health
```

**Note:** The FastAPI backend is separate from the current Streamlit Cloud deployment.

---

## ☁️ Streamlit Cloud Deployment

The current application is deployed using Streamlit Community Cloud.

### Deployment steps

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Select the UrbanPulseAI repository.
5. Select the `main` branch.
6. Select `streamlit_app.py` as the application entry point.
7. Configure the required API key under **App Settings → Secrets**.
8. Deploy the application.

### Streamlit Cloud Secrets

Add your Groq API key in the Streamlit Cloud Secrets configuration:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Use the actual environment variable name expected by your project.

Do not commit API keys to GitHub.

### Deployment Architecture

The current deployment uses:

- Streamlit Community Cloud for the application.
- LangGraph initialized directly inside Streamlit.
- Groq API for language model responses.
- Project data and retrieval components available to the deployed application.

A separate FastAPI deployment is not required for the current configuration.

---

## 📊 Data and Dataset Information

The current project uses the following data sources:

| Data Source | Description |
|---|---|
| Traffic Advisory | Stored advisory document and associated image with extraction outputs |
| Road Network | OpenStreetMap-derived GeoJSON road network data |
| Traffic Records | Synthetic dataset containing 500 traffic records |
| Weather Information | Stored Hyderabad weather snapshot |
| Processed Data | Extracted records, text chunks, metadata, and embeddings |

### Dataset Limitations

- Traffic records are synthetic and do not represent verified real-time traffic observations.
- Weather information is a stored snapshot, not a live weather feed.
- The included traffic advisory is historical and expired.
- Road network data covers a selected Gachibowli-area study region, not the entire Hyderabad road network.
- The application should not be treated as a live navigation, emergency advisory, or real-time traffic service.

---

## 🧪 Testing

The repository includes automated testing and GitHub Actions CI configuration.

Run the test suite locally:

```bash
uv run pytest
```

---

## 🔮 Future Enhancements

- Integration of verified real-time traffic APIs.
- Live weather data integration.
- Expanded Hyderabad road network coverage.
- Advanced multimodal image retrieval.
- Improved spatial and route analysis.
- Enhanced agentic reasoning and tool orchestration.
- Docker-based deployment.
- Expanded automated testing and CI/CD.
- Cloud backend integration and production monitoring.

---

## 👩‍💻 Author

**PurpleCoder-23**

GitHub: https://github.com/PurpleCoder-23

---

## 📄 License

A license has  been added to this repository.
