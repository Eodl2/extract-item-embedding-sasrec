import torch

# Load the saved item embeddings
embedding_path = "ml-1m_default/item_embeddings.pt"  # Replace with your actual path
item_embeddings = torch.load(embedding_path)

# Check the shape and content of the embeddings
print("Shape of item embeddings:", item_embeddings.shape)
print("Sample item embeddings (first 5 rows):")
print(item_embeddings[:5])
