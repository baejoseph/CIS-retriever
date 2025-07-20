# 🔍 RAG-Lite: Smart Document Q&A

A lightweight but powerful implementation of Retrieval Augmented Generation (RAG) with hierarchical document chunking and evidence-based responses.

## ✨ Key Features

### 📚 Smart Document Processing
- 🎯 Hierarchical chunking based on document structure
- 📑 Preserves section headings and document hierarchy
- 🔄 Automatic duplicate detection for multiple uploads
- 💾 Smart caching system for processed documents

### 🧠 Intelligent Retrieval
- 🎯 Dynamic similarity threshold adjustment
- 🔝 Configurable top-K retrieval
- 📊 Cosine similarity for semantic matching
- 🏷️ Section-aware chunk identification

### 🤖 Advanced Response Generation
- 📝 Evidence-based responses with source citations
- 🔍 References to specific document sections
- 🎨 Clean and intuitive chat interface
- 📈 Token estimation and optimization

### 🛠️ Technical Features
- 🔒 Type-safe implementation with dataclasses
- 🧪 Testable architecture with dependency injection
- 📦 Efficient caching and state management
- 🚀 Streamlit-powered responsive UI

## 🚀 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/rag-lite.git
   cd rag-lite
   ```

2. **Set Up Python Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   uv pip install -e .
   ```

4. **Install Local LLMs (for local running mode)**
   - Install Ollama and install necessary models:
     ```bash
     # For Inference (requires 5.2 GB)
     ollama pull deepseek-r1:latest

     # For Embedding (requires 669 MB)
     ollama pull mxbai-embed-large
     ```

5. **Set Up Environment Variables**
   ```bash
   cp env.template .env
   # Edit .env and add your OpenAI API and AWS keys
   ```

## 🎮 Usage

1. **Start the Application (app_local.py for local mode)**
   ```bash
   streamlit run rag-lite/app.py
   ```

2. **Upload Documents**
   - 📂 Click the file upload button in the sidebar
   - 📄 Upload any DOCX files (< 2MB)
   - 🔄 Watch as it's processed and chunked intelligently (caching!)

3. **Configure Retrieval**
   - 🎚️ Adjust "Top K Chunks" slider (1-10)
   - 🎯 Set "Similarity Threshold" (0.0-1.0)
   - 💡 Higher threshold = more relevant but fewer results

4. **Ask Questions**
   - ❓ Type your question in the chat input
   - 🤖 Get responses with evidence citations
   - 📚 See which sections of your documents were used

## 🎯 Example Queries

```
Q: What are the main points in the introduction?
A: Based on Section 1 (Introduction), the main points are...

Q: Summarize the methodology section.
A: According to Section 3.2 (Methodology), the approach involves...
```

## 🏗️ Project Structure

```
rag-lite/
├── app.py              # Streamlit UI and main application
├── parser.py           # Document parsing and chunking
├── rag_pipeline.py     # Core RAG implementation
├── openai_services.py  # OpenAI API integration
└── logger.py           # Logging configuration
```

## 🛠️ Advanced Configuration

### Chunking Strategy
The system uses a hierarchical chunking strategy based on document structure:
- 📑 Respects document hierarchy (chapters, sections, subsections)
- 🎯 Maintains context through section headings
- 🔍 Preserves relationships between chunks

### Retrieval Settings
- **Top K**: Number of chunks to retrieve (default: 3)
- **Similarity Threshold**: Minimum similarity score (default: 0.6)
  - 0.8-1.0: Very strict matching
  - 0.6-0.8: Balanced retrieval
  - 0.4-0.6: More exploratory results

## 📝 License

MIT License - feel free to use and modify! 