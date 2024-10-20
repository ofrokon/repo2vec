# Repo2Vec

Repo2Vec is a tool for creating vector representations (embeddings) of software repositories. It combines information from repository metadata, directory structure, and source code to create a comprehensive embedding.

## Installation

```bash
git clone https://github.com/ofrokon/repo2vec.git
cd repo2vec
pip install -r requirements.txt
```

## Usage

```python
from repo2vec import Repo2Vec

repo_path = "/path/to/your/repository"
repo2vec = Repo2Vec(repo_path)
embedding = repo2vec.generate_embedding()
print(f"Repository embedding shape: {embedding.shape}")
```

For more detailed usage examples, see the `examples` directory.

## Documentation

For detailed API documentation, see [docs/API.md](docs/API.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
