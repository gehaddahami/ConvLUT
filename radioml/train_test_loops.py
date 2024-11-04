
# Imports 
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm 
from sklearn.metrics import accuracy_score

# the functions below are different traininf and testing loops that are used baed on the classification nature whether it is binary or multicalss, or weather or not the sigmoid function is to be tested 
def train_loop_pytorch(model, train_loader, optimizer, criterion, options):
    losses = []
    model.train()

    for (inputs, labels, snr) in train_loader:  
        if options['cuda']: 
            inputs, labels = inputs.cuda(), labels.cuda()

        # Forward pass 
        output = model(inputs) 
        loss = criterion(output, labels)

        # Backward pass and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())  # Use .item() to get the scalar value of the loss

    return losses



def train_logicnets(model, train_loader, optimizer, criterion, options): 
    model.train()
    total_loss = 0.0 
    correct = 0

    for inputs, labels, snr in train_loader:
        # if options['cuda']: 
        #     inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        output = model(inputs)

        # Assuming binary classification
        loss = criterion(output, labels)
        
        # Use sigmoid for binary classification, apply threshold
        pred = (torch.sigmoid(output.detach()) > 0.5).long()
        correct += (pred == labels.unsqueeze(1)).sum().item()

        # Calculate the number of correct predictions
        # correct += pred.eq(labels.unsqueeze(1)).sum().item()

        # Accumulate loss for current batch
        total_loss += loss.item() * len(inputs)

        # Backpropagation
        loss.backward()
        optimizer.step()
    
    # Calculate average loss and accuracy for the entire dataset
    average_loss = total_loss / len(train_loader.dataset)
    accuracy = 100.0 * correct / len(train_loader.dataset)

    return average_loss, accuracy



def val_test_pytorch(model, val_loader, options):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        
        for inputs, labels, snr in val_loader:  
        
            if options['cuda']: 
                inputs, labels = inputs.cuda(), labels.cuda()
    
            outputs = model(inputs)
            # Apply softmax for multi-class classification
            softmax_outputs = torch.softmax(outputs, dim=1)

            # Take the argmax to get the predicted class
            pred = torch.argmax(softmax_outputs, dim=1).cpu().numpy()

            true_labels.append(labels.numpy())  
            predictions.append(pred)

        # printing the accuracy of the model 
        true_labels = np.concatenate(true_labels)
        predictions = np.concatenate(predictions)

    return accuracy_score(true_labels, predictions)


def binary_val_test(model, val_loader, options):  
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels, snr in val_loader:
            if options['cuda']:
                inputs, labels = inputs.cuda(), labels.cuda()
    
            outputs = model(inputs)
            # Use sigmoid activation function for binary classification
            pred = torch.round(torch.sigmoid(outputs)).reshape(-1).cpu().numpy()

            y_true.append(labels.cpu().numpy())
            y_pred.append(pred)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    # print(f'Validation Accuracy: {accuracy:.4f}')

    return accuracy


def test_logicnets(model, dataset_loader, cuda, thresh=0.5):
    model.eval()
    correct = 0
    accLoss = 0.0

    with torch.no_grad(): 
        for data, target, snr in dataset_loader:

            if cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            pred = (output.detach() > thresh) 

            curCorrect = pred.eq(target.unsqueeze(1)).long().sum()
            curAcc = 100.0 * curCorrect / len(data)

            correct += curCorrect
    testing_accuracy = 100 * float(correct) / len(dataset_loader.dataset)
    return testing_accuracy



# plotting losses and/or accuracy of the model
def display_loss(losses, title = 'Training loss', xlabel= 'Iterations', ylabel= 'Loss'):
    x_axis = [i for i in range(len(losses))] 
    plt.plot(x_axis, losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)



# Plotting function for loss and accuracy
def plot_training_results(train_losses, val_accuracies, title_loss='Training Loss', title_acc='Validation Accuracy'):
    epochs = range(1, len(train_losses) + 1)

    # Plot training loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title(title_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title(title_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()