'''
This file contains the training loop, testing loop, and the plotting functions for the loss curve and the cofusion matrix
'''
# Imports 
import os 
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm 
from sklearn.metrics import accuracy_score



def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure(figsize=(12,8), dpi=800)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def train_logicnets(model, train_loader, optimizer, criterion, options): 
    model.train()
    total_loss = 0.0 
    correct = 0

    for inputs, labels, snr in train_loader:
        if options['cuda']: 
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, labels)
        
        pred = (torch.softmax(output.detach(), dim=1) > 0.75).long()
        correct += (pred == labels.unsqueeze(1)).sum().item()

        # Accumulate loss for current batch
        total_loss += loss.item() * len(inputs)

        # Backpropagation
        loss.backward()
        optimizer.step()
    
    # Calculate average loss and accuracy for the entire dataset
    average_loss = total_loss / len(train_loader.dataset)
    accuracy = 100.0 * correct / len(train_loader.dataset)

    return average_loss, accuracy


def test_logicnets(model, val_loader, options, dataset, test=True):
    model.eval()
    true_labels = []
    predictions = []

    # Arrays for the confusion matrix
    y_exp = np.empty((0))
    y_snr = np.empty((0))
    y_pred = np.empty((0, len(dataset.mod_classes)))

    with torch.no_grad():
        for inputs, labels, snr in val_loader:
            if options['cuda']: 
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            softmax_outputs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(softmax_outputs, dim=1).cpu().numpy()

            true_labels.append(labels.cpu().numpy())  
            predictions.append(pred)

            y_pred = np.concatenate((y_pred, outputs.cpu().numpy()))
            y_exp = np.concatenate((y_exp, labels.cpu().numpy()))  
            y_snr = np.concatenate((y_snr, snr))

    # Printing the accuracy of the model 
    true_labels = np.concatenate(true_labels)
    predictions = np.concatenate(predictions)

    if test:
        conf = np.zeros([len(dataset.mod_classes), len(dataset.mod_classes)])
        confnorm = np.zeros([len(dataset.mod_classes), len(dataset.mod_classes)])

        for i in range(len(y_exp)):
            j = int(y_exp[i])
            k = int(np.argmax(y_pred[i, :]))
            conf[j, k] += 1

        for i in range(len(dataset.mod_classes)):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :]) if np.sum(conf[i, :]) > 0 else 0

        # Plotting the confusion matrix
        plot_confusion_matrix(confnorm, labels=dataset.mod_classes)
        plt.show()
        # Saving the confusion matrix, If needed then uncomment the following lines
        # plt.savefig(f"{options['log_dir']}/confusion_matrix.pdf", dpi=800, format='pdf', bbox_inches='tight')
        # plt.close()

        # Saving confusion matrix data to CSV files, if needed then uncomment the following lines
        # np.savetxt(f"{options['log_dir']}/confusion_matrix_raw.csv", conf, delimiter=",", fmt='%d')
        # np.savetxt(f"{options['log_dir']}/confusion_matrix_normalized.csv", confnorm, delimiter=",", fmt='%.4f')

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print(f"Overall Accuracy - all SNRs: {cor / (cor + ncor):.6f}")

    return accuracy_score(true_labels, predictions)


# plotting losses and/or accuracy of the model
def display_loss(losses, title = 'Training loss', xlabel= 'Iterations', ylabel= 'Loss'):
    x_axis = [i for i in range(len(losses))] 
    plt.plot(x_axis, losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# Plotting function for loss and accuracy
def plot_training_results(train_losses, val_accuracies, log_dir, plot_name = 'name.pdf', title_loss='Training Loss and Validation Accuracy'):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='orange')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='blue')

    plt.title(title_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.legend()

    plt.tight_layout()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    plt.savefig(os.path.join(log_dir, plot_name), format='pdf', dpi=800)

    plt.close()