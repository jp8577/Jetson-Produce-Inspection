from ultralytics import YOLO
import kagglehub
import os
import yaml
from IPython.display import Image, display

# Load dataset with annotations from Kaggle
path = kagglehub.dataset_download("henningheyen/lvis-fruits-and-vegetables-dataset")

# Update root path (cache file directory is hidden by default)
root_path = '../.cache/kagglehub/datasets/henningheyen/lvis-fruits-and-vegetables-dataset/versions/1'
yaml_file = '../.cache/kagglehub/datasets/henningheyen/lvis-fruits-and-vegetables-dataset/versions/1/LVIS_Fruits_And_Vegetables/data.yaml'

# Load the current configuration
with open(yaml_file, 'r') as f:
    config = yaml.safe_load(f)

# Overwrite the root path to point to the actual cache folder
config['path'] = '../.cache/kagglehub/datasets/henningheyen/lvis-fruits-and-vegetables-dataset/versions/1/LVIS_Fruits_And_Vegetables'

# Save the changes back to the file
with open(yaml_file, 'w') as f:
    yaml.dump(config, f)

print("data.yaml root path updated successfully!")

# Load a pretrained YOLOv8m model (model weights)
model = YOLO('models/yolo_fruits_and_vegetables_v1.pt')

# Test images
# image_dir = f"{root_path}/LVIS_Fruits_And_Vegetables/images/test"
image_dir = 
images = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

num_images = 20

# Set confidence and iou threshold
conf = 0.6
iou = 0.1
show_results = False

# Run batched inference on a list of images
results = model(images[:num_images], conf=conf, iou=iou)  # return a list of Results objects

if show_results:
    for result in results:
        result.show()

# create a directory to save the results
if not os.path.exists('Example_Results'):
    os.makedirs('Example_Results')

# Save results and show detections
for result in results:
    result.show()
    # save locally
    result.save(filename=f'Example_Results/{os.path.basename(result.path)}.png')

# Show images

for i in range(num_images):
    display(Image(f'Example_Results/{os.path.basename(images[i])}.png', width=200))
