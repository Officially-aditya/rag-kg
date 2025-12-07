# Project 1: Simple PDF RAG Chatbot

## Goal
Build a basic RAG system that answers questions about PDF documents.

## Skills You'll Practice
- PDF text extraction
- Text chunking
- Embeddings generation
- Vector search
- Basic prompting
- Citation formatting

## Architecture

```
PDF Files → Extract Text → Chunk into Segments → Generate Embeddings
                                                         ↓
                                                    ChromaDB Storage
                                                         ↓
User Query → Embed Query → Retrieve Top 5 Chunks → Generate Answer (with citations)
```

## Tasks

### Part 1: Setup (15 min)
1. Install required packages (see requirements.txt)
2. Set up OpenAI API key in `.env`
3. Prepare 3-5 PDF documents (research papers, documentation, etc.)

### Part 2: PDF Processing (30 min)
1. Extract text from PDFs using PyPDF2
2. Clean extracted text (remove extra spaces, page numbers)
3. Track page numbers for citations

### Part 3: Chunking (30 min)
1. Implement chunking with 500-token segments
2. Add 50-token overlap between chunks
3. Store metadata (file_name, page_number, chunk_index)

### Part 4: Embeddings & Storage (30 min)
1. Generate embeddings using `text-embedding-3-small`
2. Store chunks and embeddings in ChromaDB
3. Add metadata for filtering

### Part 5: Query System (45 min)
1. Implement query embedding
2. Retrieve top 5 most similar chunks
3. Build context from retrieved chunks
4. Generate answer using GPT-3.5-turbo or GPT-4
5. Add citations (file name + page number)

### Part 6: Testing & Refinement (30 min)
1. Test with 10 questions
2. Verify citations are accurate
3. Test "I don't know" responses when answer not in docs
4. Refine prompts for better answers

## Evaluation Criteria

- ✅ **Text Extraction** (20 pts): Correctly extracts text from all PDFs
- ✅ **Chunking** (20 pts): Reasonable chunk sizes, overlap implemented
- ✅ **Retrieval** (25 pts): Retrieves relevant chunks for queries
- ✅ **Answer Quality** (25 pts): Answers are accurate and coherent
- ✅ **Citations** (10 pts): Proper citations with source file and page
- **Total**: 100 points

**Passing Score**: 70/100

## Starter Code

See `starter_code.py` for the initial implementation template.

## Sample Questions to Test

1. "What is the main contribution of this paper?"
2. "What methodology was used?"
3. "What were the key findings?"
4. "Who are the authors?"
5. "What datasets were used?"
6. "What are the limitations mentioned?"
7. "What future work is suggested?"
8. "How does this compare to previous work?"
9. "What was the performance metric?" (test specific info retrieval)
10. "What is quantum computing?" (test "I don't know" for out-of-scope)

## Expected Output Format

```
Question: "What is the main contribution of this paper?"

Answer: "The main contribution is the development of a novel attention
mechanism that improves model performance on long sequences [Source:
attention_paper.pdf, Page 3]. This mechanism reduces computational
complexity from O(n²) to O(n log n) [Source: attention_paper.pdf, Page 5]."
```

## Tips for Success

1. **Start Simple**: Get basic extraction working before optimizing
2. **Test Incrementally**: Test each component separately
3. **Check Citations**: Verify page numbers are correct
4. **Handle Edge Cases**: Empty PDFs, malformed text, etc.
5. **Iterate on Prompts**: Refine your system prompt for better answers

## Bonus Challenges

- [ ] Add support for multiple file formats (DOCX, TXT, MD)
- [ ] Implement semantic chunking (chunk by paragraphs/sections)
- [ ] Add a simple web UI using Streamlit or Gradio
- [ ] Track and display retrieval scores
- [ ] Implement query rewriting for better retrieval

## Submission

When complete, your project should include:

1. **Code**: `main.py` with complete implementation
2. **README**: Document your approach and design decisions
3. **Sample Output**: Screenshots or text file with example Q&A
4. **Reflection**: 2-3 paragraphs on challenges and learnings

## Resources

- [PyPDF2 Documentation](https://pypdf2.readthedocs.io/)
- [ChromaDB Quickstart](https://docs.trychroma.com/getting-started)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

## Next Steps

After completing this project:
1. Review the solution code
2. Compare your approach with others
3. Move on to Project 2: Multi-Hop RAG System
4. Consider how you'd scale this to 1000s of documents

Good luck! 🚀
