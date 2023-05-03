import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import ModelEvaluation


import numpy as np
import matplotlib.pyplot as plt
import os


save_path = "./models/rembg_models/Resnet152/"
def train(model, loss_function, optimizer, epochs):
    # Determine whether to use GPU or CPU for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize variables for recording training history and best accuracy
    history = []
    best_acc = 0.0
    best_epoch = 0

    # Loop through the specified number of epochs
    for epoch in range(epochs):
        epoch_start = time.time()# Record the start time of the epoch
        print("Epoch: {}/{}".format(epoch + 1, epochs)) # Print the current epoch number

        model.train() # Set the model to training mode

        train_loss = 0.0  # Initialize the training loss
        train_acc = 0.0  # Initialize the training accuracy
        valid_loss = 0.0  # Initialize the validation loss
        valid_acc = 0.0  # Initialize the validation accuracy

        # Loop through the training data
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()# Zero the gradients


            outputs = model(inputs)# Forward pass


            loss = loss_function(outputs, labels)# Compute the loss
            loss.backward()# Backward pass

            optimizer.step() # Update the model parameters

            train_loss += loss.item() * inputs.size(0)# Record the training loss


            ret, predictions = torch.max(outputs.data, 1)# Compute the predicted classes


            correct_counts = predictions.eq(labels.data.view_as(predictions))# Count the correct predictions

            acc = torch.mean(correct_counts.type(torch.FloatTensor))# Compute the accuracy

            train_acc += acc.item() * inputs.size(0)# Record the training accuracy

        with torch.no_grad():

            #Change the model to evaluation mode
            model.eval()

            #Validation is performed with the validation set
            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)


                outputs = model(inputs)# Forward pass

                loss = loss_function(outputs, labels)# Compute the loss

                valid_loss += loss.item() * inputs.size(0)# Record the validation loss

                ret, predictions = torch.max(outputs.data, 1)# Compute the predicted classes

                correct_counts = predictions.eq(labels.data.view_as(predictions))# Count the correct predictions

                acc = torch.mean(correct_counts.type(torch.FloatTensor))# Compute the accuracy

                valid_acc += acc.item() * inputs.size(0) #Record the validation accuracy

        avg_train_loss = train_loss / train_data_size # Compute the average training loss
        avg_train_acc = train_acc / train_data_size# Compute the average training accuracy

        avg_valid_loss = valid_loss / valid_data_size# Compute the average validation loss
        avg_valid_acc = valid_acc / valid_data_size# Compute the average validation accuracy

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])# Record the training history

        # Save the model with the highest accuracy on the validation set
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path+'Resnet152.pth')
        epoch_end = time.time()# Record the end time of the

        # Display the results of each round
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

    return model, history


def test(model, loss_function,model_path,data):
    model = model.to('cuda')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path))# Load saved model state from the specified path

    test_loss = 0.0
    test_acc = 0.0
    test_start = time.time()
    # Set the model to evaluation mode
    with torch.no_grad():
        model.eval()

    # Iterate over the test data
    for j, (inputs, labels) in enumerate(data):
        # Move the data to the device (GPU if available)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass through the network
        outputs = model(inputs)

        # Calculate the loss between the predicted and actual labels
        loss = loss_function(outputs, labels)
        # Compute the test loss
        test_loss += loss.item() * inputs.size(0)
        # Get the predicted labels as the class with highest probability
        ret, predictions = torch.max(outputs.data, 1)
        # Count the number of correctly predicted labels
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        # Calculate the accuracy on this batch
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        # Accumulate the test accuracy over all batches
        test_acc += acc.item() * inputs.size(0)

    # Compute the average test loss and accuracy over all batches
    avg_test_loss = test_loss / test_data_size
    avg_test_acc = test_acc / test_data_size
    test_end = time.time()

    # Print the test results
    print(
        "test: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            avg_test_loss, avg_test_acc * 100,
                           test_end - test_start
        ))


seed = 1024
np.random.seed(seed)
#Fixed parameters ensure that the experiment can be redone
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#Define how the image should be processed
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),  # Centre cropping to 224*224 meets the input requirements of resnet
        transforms.ToTensor(),  # 填充
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),

}

#dataset path
dataset = 'C:/Users/20786/PycharmProjects/bg/re_dataset'

#dataset = './spli_dataset'
train_dataset = os.path.join(dataset, 'train')  # The path of the train dataset
valid_dataset = os.path.join(dataset, 'valid')  # The path of the valid dataset
test_dataset = os.path.join(dataset, 'test')  # The path of the test dataset
batch_size = 16
num_classes = 5

#Process the data
data = {
    'train': datasets.ImageFolder(root=train_dataset, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_dataset, transform=image_transforms['train']),
    'test': datasets.ImageFolder(root=test_dataset, transform=image_transforms['train'])
}

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])

#Load the dataset and shuffle it
train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
# DataLoader(dataset, batch_size, shuffle) dataset data type; number of groups; whether to shuffle
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)

print("Number of train dataset: {}，Number of valid dataset: {},Number of test dataset: {}".format(train_data_size, valid_data_size, test_data_size))

print("Data loaded successfully")


#Define the resnet model
resnet = models.resnet152(pretrained=True)  # Open pre-training
# Since most of the parameters in the pre-trained model
# have already been trained, reset the requires_grad field to false.
for param in resnet.parameters():
    param.requires_grad = False  # False: freeze the parameters of the model.

fc_inputs = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),  # Activation functions
    nn.Dropout(0.4),  # Dropout is to prevent overfitting
    nn.Linear(256, num_classes),
    nn.LogSoftmax(dim=1)
)

# Define the loss function and optimizer.
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet.parameters())


#Training switch
is_train = False

if is_train:
    #Iteration round number
    num_epochs = 10
    trained_model, history = train(resnet, loss_func, optimizer, num_epochs)


#Show switch
is_picshow = False
if is_picshow:
    #Show how the loss changes during training
    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(save_path+'loss_curve.png')
    plt.show()

    # Show how the accuracy changes during training
    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(save_path+'accuracy_curve.png')
    plt.show()

#Test switch
is_test = True
model_path= save_path+'Resnet152.pth'
if is_test:
    print()
    test(resnet, loss_func,model_path,test_data)

#Model evaluation switch
is_eva = True
if is_eva:
    ModelEvaluation.model_evaluation(resnet,model_path,test_data,save_path)
