import torch
from torch import nn
from torchvision import models
from collections import OrderedDict

"""
These functions provide functionality to load and build the model of the neural net.
"""

def build_model(architecture, hidden_units, output_units, device):
    model = eval(architecture)
    for param in model.parameters():
        param.requires_grad = False
    # consider different architectures
    try:
        input_units = model.classifier[-1].in_features
    except TypeError:
        input_units = model.classifier.in_features

    new_classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(hidden_units, output_units)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    # consider different architectures
    try:
        model.classifier[-1] = new_classifier
    except TypeError:
        model.classifier = new_classifier
  
    model = model.to(device)
    return model, input_units

def load_checkpoint(filepath, device):
    # check if GPU is available
    if device == 'cuda':
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    checkpoint = torch.load(filepath, map_location=map_location)
    model, _ = build_model(checkpoint['architecture'], checkpoint['hidden_units'], checkpoint['output_units'], device)
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['epochs'], checkpoint['class_to_idx'], checkpoint['idx_to_class']
