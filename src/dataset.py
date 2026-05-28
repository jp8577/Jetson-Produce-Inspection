import kagglehub

# Download latest version
path = kagglehub.dataset_download("ulnnproject/food-freshness-dataset")

print("Path to dataset files:", path)
