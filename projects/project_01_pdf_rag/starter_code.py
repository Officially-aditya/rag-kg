"""
Project 1: Simple PDF RAG Chatbot - Starter Code
Complete the TODOs to build your first RAG system!
"""

import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="pdf_documents")


class PDFProcessor:
    """Handles PDF extraction and chunking"""

    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF with page numbers

        TODO: Implement PDF text extraction
        - Use PyPDF2 to read PDF
        - Extract text from each page
        - Return list of dicts with: {page_number, text}
        """
        pages = []

        # YOUR CODE HERE
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                pages.append({
                    'page_number': page_num + 1,
                    'text': text
                })

        return pages

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text

        TODO: Implement text cleaning
        - Remove extra whitespace
        - Remove page numbers (if any)
        - Fix broken words
        """
        import re

        # YOUR CODE HERE
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove standalone page numbers (simple heuristic)
        text = re.sub(r'\n\d+\n', '\n', text)

        return text.strip()

    def chunk_text(self, pages: List[Dict], file_name: str) -> List[Dict]:
        """
        Chunk text with overlap and metadata

        TODO: Implement chunking
        - Split text into chunk_size segments
        - Add overlap between chunks
        - Track source page for each chunk
        """
        chunks = []

        # YOUR CODE HERE
        # Hint: You can use simple character-based chunking
        # or use LangChain's RecursiveCharacterTextSplitter

        # Simple implementation:
        for page in pages:
            text = self.clean_text(page['text'])
            start = 0

            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]

                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        'file_name': file_name,
                        'page_number': page['page_number'],
                        'char_start': start
                    }
                })

                start = end - self.overlap

        return chunks


class EmbeddingManager:
    """Handles embedding generation and storage"""

    def __init__(self):
        self.model = "text-embedding-3-small"

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text

        TODO: Implement embedding generation
        - Use OpenAI's embedding API
        - Return the embedding vector
        """
        # YOUR CODE HERE
        response = client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def store_chunks(self, chunks: List[Dict]):
        """
        Store chunks with embeddings in ChromaDB

        TODO: Implement storage
        - Generate embeddings for each chunk
        - Store in ChromaDB with metadata
        """
        # YOUR CODE HERE
        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk['content'])

            collection.add(
                documents=[chunk['content']],
                embeddings=[embedding],
                metadatas=[chunk['metadata']],
                ids=[f"chunk_{i}"]
            )


class RAGQueryEngine:
    """Handles query processing and answer generation"""

    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def retrieve(self, query: str, top_k=5) -> List[Dict]:
        """
        Retrieve most relevant chunks for query

        TODO: Implement retrieval
        - Embed the query
        - Search ChromaDB for similar chunks
        - Return top_k results
        """
        # YOUR CODE HERE
        query_embedding = self.embedding_manager.get_embedding(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        retrieved = []
        for i in range(len(results['documents'][0])):
            retrieved.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })

        return retrieved

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate answer using LLM with retrieved context

        TODO: Implement answer generation
        - Build context from chunks
        - Create prompt with instructions
        - Call OpenAI API
        - Format answer with citations
        """
        # YOUR CODE HERE
        # Build context
        context = ""
        for i, chunk in enumerate(context_chunks):
            file_name = chunk['metadata']['file_name']
            page = chunk['metadata']['page_number']
            content = chunk['content']
            context += f"\n[Source {i+1}: {file_name}, Page {page}]\n{content}\n"

        # Create prompt
        prompt = f"""
        Answer the question using only the provided context.

        IMPORTANT RULES:
        - Only use information from the context
        - If the answer is not in the context, say "I don't have enough information to answer this question."
        - Cite your sources using [Source X] notation
        - Be concise and accurate

        Context:
        {context}

        Question: {query}

        Answer:
        """

        # Generate answer
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content

    def query(self, question: str) -> Dict:
        """
        Complete query pipeline

        TODO: Tie it all together
        - Retrieve relevant chunks
        - Generate answer
        - Return formatted result
        """
        # YOUR CODE HERE
        chunks = self.retrieve(question, top_k=5)
        answer = self.generate_answer(question, chunks)

        return {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'file': chunk['metadata']['file_name'],
                    'page': chunk['metadata']['page_number']
                }
                for chunk in chunks
            ]
        }


def main():
    """Main execution function"""

    # Step 1: Process PDFs
    print("Processing PDFs...")
    processor = PDFProcessor(chunk_size=500, overlap=50)

    # TODO: Update with your PDF file paths
    pdf_files = [
        "sample_data/document1.pdf",
        "sample_data/document2.pdf",
        "sample_data/document3.pdf"
    ]

    all_chunks = []
    for pdf_path in pdf_files:
        if not Path(pdf_path).exists():
            print(f"Warning: {pdf_path} not found, skipping...")
            continue

        print(f"Processing {pdf_path}...")
        pages = processor.extract_text(pdf_path)
        chunks = processor.chunk_text(pages, Path(pdf_path).name)
        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")

    # Step 2: Generate and store embeddings
    print("\nGenerating embeddings...")
    embedding_manager = EmbeddingManager()
    embedding_manager.store_chunks(all_chunks)
    print("Embeddings stored!")

    # Step 3: Query the system
    print("\nInitializing query engine...")
    query_engine = RAGQueryEngine()

    # Example queries
    test_questions = [
        "What is the main topic of the documents?",
        "Who are the authors?",
        "What methodology was used?"
    ]

    print("\n" + "="*60)
    print("Testing RAG System")
    print("="*60)

    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = query_engine.query(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 60)

    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*60)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            continue

        result = query_engine.query(question)
        print(f"\nAnswer: {result['answer']}\n")
        print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    main()
