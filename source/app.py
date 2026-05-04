from source.data_loader import load_all_documents
from source.vectorstore import FaissVectorStore
from source.search import RAGSearch
from source.embedding import EmbeddingPipeline



# Example usage

if __name__ == "__main__":
    docs=load_all_documents("data")
    chunk=EmbeddingPipeline().chunk_documents(docs)
    chunkvectors=EmbeddingPipeline().chunk_documents(chunks)
    print(chunkvectors)

