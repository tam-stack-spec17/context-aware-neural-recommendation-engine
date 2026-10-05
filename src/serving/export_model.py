import os
import json
import numpy as np
import tensorflow as tf

class QueryServingModule(tf.Module):
    """Production serving wrapper to output dense query embeddings from raw User IDs."""
    def __init__(self, query_tower):
        super().__init__()
        self.query_tower = query_tower

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="customer_id")])
    def __call__(self, customer_id: tf.Tensor):
        return self.query_tower({"customer_id": customer_id})

def export_candidate_embeddings(candidate_tower, article_ids: list, export_path: str):
    """Batch computes and exports pre-calculated candidate vectors for Redis / ANN store."""
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    item_tensor = tf.constant(article_ids, dtype=tf.string)
    vectors = candidate_tower({"article_id": item_tensor}).numpy()
    
    candidate_map = {
        art_id: vec.tolist() for art_id, vec in zip(article_ids, vectors)
    }
    with open(export_path, "w") as f:
        json.dump(candidate_map, f)
    print(f"Exported {len(candidate_map)} item candidate vectors to {export_path}")

if __name__ == "__main__":
    print("Export module initialized for Query Tower serving signatures and item vectors.")