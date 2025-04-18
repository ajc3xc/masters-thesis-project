import os
import requests
from PIL import Image
from io import BytesIO

# Directory to save images
os.makedirs('Set5', exist_ok=True)

# List of image names
image_names = ['baby', 'bird', 'butterfly', 'head', 'woman']

# Base URL for the images
base_url = 'http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12/'

for name in image_names:
    url = f'{base_url}{name}.png'
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save(f'Set5/{name}.png')
        print(f'Downloaded and saved {name}.png')
    else:
        print(f'Failed to download {name}.png')
