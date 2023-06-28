import os
import clip
import torch

## PAOT
TASK_NAME = ['Scattering correction',
             'removing circular artifacts',
             'converting low-energy images to high-energy images',
             'temporal CT',
             'MRI image conversion']

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

text_inputs = torch.cat([clip.tokenize(f'A CT image processed through {item} process') for item in TASK_NAME]).to(device)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, 'txt_encoding.pth')
