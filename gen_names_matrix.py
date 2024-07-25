from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import torch
import numpy as np
import time

start_time = time.time()

names = []
with open('names.txt', 'r') as file:
    for line in file:
        name = line.strip()
        if name:
            names.append(name)

bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3', 
    device='cpu',
    return_sparse=False
)

name_embeddings_matrix = np.array(bge_m3_ef.encode_documents(names)["dense"]).T  # Transpose to make each embedding a column

np.save('names.npy', name_embeddings_matrix)
print(f"Matrix shape: {name_embeddings_matrix.shape}")
print("Matrix saved as names.npy")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")