import kagglehub

# Download latest version
path = kagglehub.dataset_download("henningheyen/lvis-fruits-and-vegetables-dataset")

print("Path to dataset files:", path)
