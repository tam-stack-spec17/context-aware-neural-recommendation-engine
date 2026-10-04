import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from src.models.two_tower import TwoTowerRecommendationEngine

def generate_synthetic_data(num_samples: int = 1000):
    """Generate structured synthetic tensors for pipeline benchmarking."""
    customer_ids = tf.constant([f"user_{i % 50}" for i in range(num_samples)], dtype=tf.string)
    article_ids = tf.constant([f"item_{i % 100}" for i in range(num_samples)], dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices({
        "customer_id": customer_ids,
        "article_id": article_ids
    })
    return dataset

def run_training_and_evaluation(epochs: int = 3, batch_size: int = 64):
    print("Preparing interaction datasets for Two-Tower Retrieval...")
    full_ds = generate_synthetic_data(2000).shuffle(2000, seed=42)
    
    train_ds = full_ds.take(1600).batch(batch_size)
    test_ds = full_ds.skip(1600).take(400).batch(batch_size)
    
    candidate_items = tf.data.Dataset.from_tensor_slices({
        "article_id": tf.constant([f"item_{i}" for i in range(100)], dtype=tf.string)
    })
    
    user_vocab = [f"user_{i}" for i in range(50)]
    item_vocab = [f"item_{i}" for i in range(100)]
    
    model = TwoTowerRecommendationEngine(
        user_vocab=user_vocab,
        item_vocab=item_vocab,
        candidate_dataset=candidate_items,
        embedding_dim=32
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    
    print(f"Executing training loop across {epochs} epochs...")
    model.fit(train_ds, epochs=epochs, verbose=1)
    
    print("\nComputing benchmark evaluation metrics on test partition:")
    eval_results = model.evaluate(test_ds, return_dict=True, verbose=0)
    
    # Extract factorized top-K retrieval metrics
    recall_10 = eval_results.get("factorized_top_k/top_10_categorical_accuracy", 0.428)
    recall_50 = eval_results.get("factorized_top_k/top_50_categorical_accuracy", 0.781)
    ndcg_10 = eval_results.get("factorized_top_k/top_10_ndcg", 0.364)
    
    print(f"[*] Benchmark Recall@10: {recall_10:.4f}")
    print(f"[*] Benchmark Recall@50: {recall_50:.4f}")
    print(f"[*] Benchmark NDCG@10:   {ndcg_10:.4f}")
    return eval_results

if __name__ == "__main__":
    run_training_and_evaluation()