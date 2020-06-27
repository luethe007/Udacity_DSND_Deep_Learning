# Imports
import torch
from torch import nn
import argparse
from data import load_data
import build
from torch import optim

# define parameters
save_dir = "checkpoint.pth"
arch = "models.densenet121(pretrained=True)"
learning_rate = 0.003
hidden_units = 512
output_units = 102
epochs = 1
device = 'cpu'

"""
This script trains, tests and saves a neural net. Hyperparameters can be passed via command line.
"""

def train_network():
    global device
    trainloader, validloader, testloader, train_data = load_data(train_dir, valid_dir, test_dir)
    model, input_units = build.build_model(arch, hidden_units, output_units, device)
    steps = 0
    running_loss = 0
    print_every = 5
    criterion = nn.NLLLoss()
    try: 
        optimizer = optim.Adam(model.classifier[-1].parameters(), learning_rate)
    except ValueError: 
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
        
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            if device == 'cuda' and torch.cuda.is_available():
                device = 'cuda'
            else: 
                device = 'cpu'
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                    
    return model, testloader, train_data

def test_model(model, testloader):
    global device
    test_loss = 0
    accuracy = 0
    model.eval()
    criterion = nn.NLLLoss()
    with torch.no_grad():
        for inputs, labels in testloader:
            if device == 'cuda' and torch.cuda.is_available():
                device = 'cuda'
            else: 
                device = 'cpu'
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")
    
def save_model(model, train_data):
    class_to_idx = train_data.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    checkpoint = {'architecture': arch,
                  'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': class_to_idx,
                  'idx_to_class': idx_to_class,
                  'hidden_units': hidden_units,
                  'output_units': output_units
                 }

    torch.save(checkpoint, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="data directory")
    parser.add_argument("--save_dir", help="save directory")
    parser.add_argument("--arch", help="neural net architecture")
    parser.add_argument("--learning_rate", help="learning rate")
    parser.add_argument("--hidden_units", help="hidden units")
    parser.add_argument("--epochs", help="epochs")
    parser.add_argument("--gpu", help="use gpu for training", action='store_true')

    args = parser.parse_args()
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
   
    if args.save_dir is not None:
        save_dir = args.save_dir
    if args.arch is not None:
        arch = 'models.'+ str(args.arch) + '(pretrained=True)'
    if args.learning_rate is not None:
        learning_rate = float(args.learning_rate)
    if args.hidden_units is not None:
        hidden_units = int(args.hidden_units)
    if args.epochs is not None:
        epochs = int(args.epochs)
    if args.gpu or torch.cuda.is_available():
        device = 'cuda'
    
    model, testloader, train_data = train_network()
    test_model(model, testloader)
    save_model(model, train_data)