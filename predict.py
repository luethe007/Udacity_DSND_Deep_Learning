import argparse
import build
from PIL import Image
import torch
import numpy as np
import json

# define parameters
top_k = 1
category_names = "cat_to_name.json"
device = 'cpu'

"""
This script preprocesses images and predicts the most probable class.
"""

def process_image():
    # load image
    im = Image.open(image_path)

    # resize image
    original_size = im.size
    desired_size = 257
    ratio = float(desired_size)/min(original_size)
    new_size = tuple([int(x*ratio) for x in original_size])
    im.thumbnail(new_size, Image.ANTIALIAS)

    # crop image
    width = new_size[0]
    height = new_size[1]
    cropped_im = im.crop((width//2 - 224//2, height//2 - 224//2, width//2 + 224//2, height//2 + 224//2))

    # retrieve numpy array
    np_image = np.array(cropped_im)/255.0
    np_array = ((np_image - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]).transpose(2,0,1)
    return torch.from_numpy(np_array)

def predict():
    global device
    model, epochs, class_to_idx, idx_to_class = build.load_checkpoint(checkpoint, device)
    test_image = process_image().unsqueeze_(0).float()
    model.eval()
    
    if device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
    
    model = model.to(device)
    test_image = test_image.to(device)
    with torch.no_grad():
        logps = model.forward(test_image)

    ps = torch.exp(logps)
    top_p, top_class = ps.topk(top_k, dim = 1)
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    if len(top_class[0]) > 1:
        print('Predicted Flowers:')
    else:
        print('Predicted Flower:')
    for i in range(len(top_class[0])):
        print('Flower ' + str(i+1) + ': ' + str(cat_to_name[idx_to_class[top_class[0][i].item()]]) + \
              ' with Probability ' + str(round(top_p[0][i].item()*100,2)) + '%')
              
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="path to image")
    parser.add_argument("checkpoint", help="last checkpoint")
    parser.add_argument("--top_k", help="return top k most likely classes")
    parser.add_argument("--category_names", help="mapping of categories to real names")
    parser.add_argument("--gpu", help="use gpu for inference", action='store_true')

    args = parser.parse_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    if args.top_k is not None:
        top_k = int(args.top_k)
    if args.category_names is not None:
        category_names = args.category_names
    if args.gpu or torch.cuda.is_available():
        device = 'cuda'
        
    predict()
        