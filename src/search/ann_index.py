import numpy as np
from typing import List, Tuple, Dict

class VectorANNIndex:
    """Fast cosine-similarity Approximate Nearest Neighbor index for candidate items."""
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.article_ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None

    def build_index(self, item_vectors: Dict[str, List[float]]):
        """Constructs normalized vector matrix for rapid dot-product retrieval."""
        self.article_ids = list(item_vectors.keys())
        raw_matrix = np.array(list(item_vectors.values()), dtype=np.float32)
        
        # L2-normalize vectors for fast cosine distance via dot product
        norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.vectors = raw_matrix / norms
        print(f"Indexed {len(self.article_ids)} candidate items into ANN search space.")

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieves top-K nearest candidate article IDs ranked by similarity score."""
        if self.vectors is None or len(self.article_ids) == 0:
            raise ValueError("ANN Index is empty. Call build_index first.")

        # L2-normalize query vector
        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        
        # High-speed matrix dot product against all candidate vectors
        scores = np.dot(self.vectors, q_norm)
        
        # Partition top-K candidates
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        sorted_indices = top_indices[np.argsort(-scores[top_indices])]
        
        return [(self.article_ids[idx], float(scores[idx])) for idx in sorted_indices]

if __name__ == "__main__":
    index = VectorANNIndex(embedding_dim=64)
    # Validate with synthetic vector space
    sample_data = {f"item_{i}": np.random.randn(64).tolist() for i in range(500)}
    index.build_index(sample_data)
    
    test_query = np.random.randn(64)
    results = index.search(test_query, top_k=5)
    print("Top-5 ANN Search Results:")
    for item_id, score in results:
        print(f"  {item_id} -> similarity: {score:.4f}")