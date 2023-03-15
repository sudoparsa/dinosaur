import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def data(X, y, path):
    for i, img in enumerate(tqdm(X)):
        im = Image.fromarray(((img)*255).astype(np.uint8))
        im.save(os.path.join(path, str(y[i]), str(i)+'.jpg'))

folder = 'waterbirds'
images = np.load(open(os.path.join(folder, "data64.npy"), "rb"))
images = np.transpose(images, [0,2,3,1])
label = np.load(open(os.path.join(folder, "labels.npy"), "rb"))

folder = 'waterbirds64'
os.mkdir(folder)
os.mkdir(os.path.join(folder, 'train'))
os.mkdir(os.path.join(folder, 'val'))
for i in range(2):
    os.mkdir(os.path.join(folder, 'train', str(i)))
for i in range(2):
    os.mkdir(os.path.join(folder, 'val', str(i)))

data(images, label, os.path.join(folder, 'train'))
# data(X_val, y_val, os.path.join(folder, 'val'))
