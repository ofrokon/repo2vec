# Repo2Vec

Repo2Vec is a powerful tool for creating vector representations (embeddings) of software repositories. It combines information from repository metadata, directory structure, and source code to create a comprehensive embedding that can be used for various analysis tasks such as repository similarity comparison, clustering, and classification.

This implementation is based on the work described in the following paper:

Rokon, M. O. F., Yan, P., Islam, R., & Faloutsos, M. (2021). Repo2Vec: A Comprehensive Embedding Approach for Determining Repository Similarity. arXiv preprint arXiv:2107.05112.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [API Documentation](#api-documentation)
4. [How It Works](#how-it-works)
5. [Citation](#citation)
6. [Contributing](#contributing)
7. [License](#license)

## Installation

To install Repo2Vec, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/repo2vec.git
   ```

2. Change to the project directory:
   ```
   cd repo2vec
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install the package:
   ```
   pip install .
   ```

## Usage

Here's a basic example of how to use Repo2Vec:

```python
from repo2vec import Repo2Vec

# Initialize Repo2Vec with the path to your repository
repo_path = "/path/to/your/repository"
repo2vec = Repo2Vec(repo_path)

# Generate the embedding
embedding = repo2vec.generate_embedding()

print(f"Repository embedding shape: {embedding.shape}")
print(f"Repository embedding: {embedding}")
```

For more detailed examples, see the `examples/example_usage.py` file in the repository.

## API Documentation

### Repo2Vec

The main class for generating repository embeddings.

#### `__init__(self, repo_path, vector_size=128, weights=(1, 1, 1))`

Constructor for the Repo2Vec class.

- `repo_path` (str): Path to the repository you want to analyze.
- `vector_size` (int, optional): Size of the embedding vector for each component (meta, struct, source). Default is 128.
- `weights` (tuple, optional): Weights for combining the meta, struct, and source vectors. Default is (1, 1, 1).

#### `generate_embedding(self)`

Generates the embedding for the repository.

- Returns: numpy.ndarray: The combined embedding vector for the repository.

### Meta2Vec

Class for generating embeddings from repository metadata.

#### `__init__(self, repo_path, vector_size)`

Constructor for the Meta2Vec class.

- `repo_path` (str): Path to the repository.
- `vector_size` (int): Size of the embedding vector.

#### `generate(self)`

Generates the metadata embedding.

- Returns: numpy.ndarray: The metadata embedding vector.

### Struct2Vec

Class for generating embeddings from repository structure.

#### `__init__(self, repo_path, vector_size)`

Constructor for the Struct2Vec class.

- `repo_path` (str): Path to the repository.
- `vector_size` (int): Size of the embedding vector.

#### `generate(self)`

Generates the structure embedding.

- Returns: numpy.ndarray: The structure embedding vector.

### Source2Vec

Class for generating embeddings from repository source code.

#### `__init__(self, repo_path, vector_size)`

Constructor for the Source2Vec class.

- `repo_path` (str): Path to the repository.
- `vector_size` (int): Size of the embedding vector.

#### `generate(self)`

Generates the source code embedding.

- Returns: numpy.ndarray: The source code embedding vector.

## How It Works

Repo2Vec generates embeddings using three main components:

1. **Meta2Vec**: Analyzes repository metadata (README, description, etc.) using Doc2Vec.
2. **Struct2Vec**: Represents the directory structure as a graph and generates an embedding using Node2Vec.
3. **Source2Vec**: Analyzes Java source code using a pre-trained Code2Vec model.

These embeddings are then combined into a single vector representation of the repository.

## Citation

If you use Repo2Vec in your research or project, please cite the original paper:

```
@inproceedings{rokon2021repo2vec,
  title={Repo2vec: A comprehensive embedding approach for determining repository similarity},
  author={Rokon, Md Omar Faruk and Yan, Pei and Islam, Risul and Faloutsos, Michalis},
  booktitle={2021 IEEE International Conference on Software Maintenance and Evolution (ICSME)},
  pages={355--365},
  year={2021},
  organization={IEEE}
}
```

## Contributing

Contributions to Repo2Vec are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

Please ensure that your code adheres to the existing style and that you have added appropriate tests for your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
