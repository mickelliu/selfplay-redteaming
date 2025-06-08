from sentence_transformers import SentenceTransformer
import torch

def compute_batch_sbert_similarity(sentences: list[str], model=None) -> float:
    # Load the model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # Compute embeddings
    embeddings = model.encode(sentences)
    embeddings = torch.tensor(embeddings)

    # Compute similarity matrix
    similarities = model.similarity(embeddings, embeddings)

    # Compute average similarity (excluding self-similarity)
    # Create a mask to exclude diagonal elements (self-similarity)
    mask = ~torch.eye(similarities.shape[0], dtype=bool)
    avg_similarity = similarities[mask].mean()

    return avg_similarity


if __name__ == "__main__":
    # Load the model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Your batch of sentences
    sentences = [
        "It's so cold outside!",
        "It's so hot outside!"
    ]

    avg_similarity = compute_batch_sbert_similarity(sentences, model)
    print(f"Average in-batch similarity: {avg_similarity}")
