# Delete existing model files
import os
import glob

# Find and remove all model files
model_files = glob.glob('c:/Users/Karan/Desktop/EasyML/models/*.pt')
for file in model_files:
    os.remove(file)
    print(f"Deleted: {file}")