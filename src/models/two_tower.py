import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import List

class UserQueryTower(tf.keras.Model):
    """Query Tower: Maps user identifiers and demographics to a dense representation."""
    def __init__(self, user_vocab: List[str], embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=user_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(user_vocab) + 1, embedding_dim)
        ])
        self.dense_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(embedding_dim)
        ])

    def call(self, inputs: dict) -> tf.Tensor:
        user_vector = self.user_embedding(inputs["customer_id"])
        return self.dense_projection(user_vector)

class ItemCandidateTower(tf.keras.Model):
    """Candidate Tower: Maps article IDs and item context to a matching dense representation."""
    def __init__(self, item_vocab: List[str], embedding_dim: int = 64):
        super().__init__()
        self.item_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=item_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(item_vocab) + 1, embedding_dim)
        ])
        self.dense_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(embedding_dim)
        ])

    def call(self, inputs: dict) -> tf.Tensor:
        item_vector = self.item_embedding(inputs["article_id"])
        return self.dense_projection(item_vector)

class TwoTowerRecommendationEngine(tfrs.Model):
    """Unified Two-Tower neural model optimizing in-batch negative sampling retrieval."""
    def __init__(self, user_vocab: List[str], item_vocab: List[str], candidate_dataset: tf.data.Dataset, embedding_dim: int = 64):
        super().__init__()
        self.query_tower = UserQueryTower(user_vocab, embedding_dim)
        self.candidate_tower = ItemCandidateTower(item_vocab, embedding_dim)

        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_dataset.batch(128).map(self.candidate_tower)
            )
        )

    def compute_loss(self, features: dict, training: bool = False) -> tf.Tensor:
        query_embeddings = self.query_tower({"customer_id": features["customer_id"]})
        candidate_embeddings = self.candidate_tower({"article_id": features["article_id"]})
        return self.retrieval_task(
            query_embeddings,
            candidate_embeddings,
            compute_metrics=not training
        )

if __name__ == "__main__":
    print("Two-Tower Neural Retrieval Architecture initialized successfully.")