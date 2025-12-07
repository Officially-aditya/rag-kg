# RAG + Knowledge Graph Master Course

**From Beginner to Hire-Ready Enterprise AI Engineer**

## Overview

This comprehensive course transforms you from a beginner into a production-ready RAG + Knowledge Graph engineer. Master the skills needed to build enterprise-grade hybrid retrieval systems used in big-tech and Fortune 500 companies.

## What You'll Learn

- ✅ Build sophisticated RAG (Retrieval-Augmented Generation) systems
- ✅ Design and query knowledge graphs with Neo4j
- ✅ Create hybrid architectures combining RAG + KG
- ✅ Deploy production-ready AI applications
- ✅ Evaluate and optimize retrieval systems

## Course Structure

### 📚 Main Curriculum
- **Module 1**: Beginner-Friendly Foundations (LLMs, Embeddings, Graph Theory)
- **Module 2**: RAG Engineering (Chunking, Retrieval, Reranking)
- **Module 3**: Knowledge Graph Engineering (Schema Design, Cypher, Graph Traversal)
- **Module 4**: Hybrid RAG + KG Systems (Query Routing, Context Fusion)
- **Module 5**: Practical Engineering (Deployment, Evaluation, Optimization)

### 🛠️ Projects
10 hands-on projects + 1 capstone, progressively building from:
1. Simple PDF RAG Chatbot
2. Multi-Hop RAG System
3. Automatic KG Builder
4. Entity/Relationship Extractor
5. Cypher Query Generator
6. KG Search Engine
7. RAG with Reranker
8. Graph-RAG with Neighborhood Expansion
9. Hybrid RAG + KG Chatbot
10. Production-Ready Enterprise Knowledge Assistant

**Capstone**: Enterprise "Company Brain" - A complete hybrid system

## Prerequisites

- Basic Python programming
- Familiarity with command line
- Understanding of basic ML concepts (helpful but not required)

## Tech Stack

- **Languages**: Python 3.11+
- **LLM APIs**: OpenAI (GPT-4, GPT-3.5-turbo)
- **Embeddings**: OpenAI embeddings, sentence-transformers
- **Vector DBs**: FAISS, ChromaDB, Pinecone
- **Graph DB**: Neo4j
- **Frameworks**: LangChain, LlamaIndex
- **Backend**: FastAPI
- **Deployment**: Docker, Docker Compose
- **Testing**: pytest, RAGAS, TruLens

## Getting Started

### 1. Installation

```bash
# Clone or download the course materials
cd rag+kg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Setup Neo4j

```bash
# Using Docker
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

Access Neo4j Browser at: http://localhost:7474

### 3. Configure API Keys

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### 4. Start Learning

Open `RAG_KG_Master_Course.md` and begin with Module 1!

## Project Structure

```
rag+kg/
├── RAG_KG_Master_Course.md       # Main course content
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── projects/                      # Project folders
│   ├── project_01_pdf_rag/
│   │   ├── README.md
│   │   ├── starter_code.py
│   │   └── sample_data/
│   ├── project_02_multihop_rag/
│   ├── ...
│   └── capstone_company_brain/
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_embeddings_intro.ipynb
│   ├── 02_rag_basics.ipynb
│   └── 03_kg_basics.ipynb
└── utils/                         # Shared utility code
    ├── __init__.py
    ├── embeddings.py
    ├── chunking.py
    └── graph_utils.py
```

## Learning Path

1. **Week 1-2**: Foundations + RAG Engineering
   - Study Modules 1-2
   - Complete Projects 1-2

2. **Week 3-4**: Knowledge Graphs
   - Study Module 3
   - Complete Projects 3-6

3. **Week 5-6**: Hybrid Systems
   - Study Module 4
   - Complete Projects 7-9

4. **Week 7-8**: Production & Deployment
   - Study Module 5
   - Complete Project 10

5. **Week 9-12**: Capstone Project
   - Build Enterprise "Company Brain"
   - Deploy and document

## Evaluation & Certification

- **Quizzes**: Pass module quizzes with 80%+ score
- **Projects**: Complete all 10 projects + capstone
- **Portfolio**: GitHub repo + demo video
- **Blog Posts**: Write 3 technical posts explaining concepts

## Resources

### Documentation
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [LangChain Docs](https://python.langchain.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)

### Communities
- Discord: RAG/LLM engineering communities
- Twitter: #RAG, #KnowledgeGraphs, #LLMs
- GitHub: Explore open-source RAG projects

### Papers to Read
- "Retrieval-Augmented Generation for Knowledge-Intensive Tasks"
- "GraphRAG: Unlocking LLM discovery on narrative private data" (Microsoft)
- "Dense Passage Retrieval for Open-Domain Question Answering"

## Career Outcomes

Graduates of this course are qualified for:

- **RAG Engineer**: $120k-$180k
- **Knowledge Graph Engineer**: $130k-$190k
- **AI/ML Engineer (LLM focus)**: $140k-$220k
- **Senior/Staff positions**: $200k-$300k+

**Companies Hiring**: Google, Microsoft, Meta, Amazon, Anthropic, OpenAI, and more.

## Export as PDF

To generate a PDF version:

```bash
pandoc RAG_KG_Master_Course.md \
  -o RAG_KG_Master_Course.pdf \
  --pdf-engine=xelatex \
  --toc \
  --toc-depth=3 \
  --number-sections
```

## Contributing

Found an issue or want to improve the course? Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

Educational Use - Free for personal learning

## Support

- **Issues**: Open GitHub issues for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: [Your contact email]

---

**Course Version**: 1.0 (2025)
**Last Updated**: December 2025

Happy Learning! 🚀
