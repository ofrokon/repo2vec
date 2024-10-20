# examples/example_usage.py

from repo2vec import Repo2Vec

def main():
    # Path to the repository you want to analyze
    repo_path = "/path/to/your/repository"
    
    # Initialize Repo2Vec
    repo2vec = Repo2Vec(repo_path, vector_size=128, weights=(1, 0.8, 1.2))
    
    # Generate the embedding
    embedding = repo2vec.generate_embedding()
    
    if embedding is not None:
        print(f"Repository embedding shape: {embedding.shape}")
        print(f"Repository embedding: {embedding}")
    else:
        print("Failed to generate repository embedding")

if __name__ == "__main__":
    main()
