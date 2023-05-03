import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# Define a function for predicting the image class using the given model and image
def pre(model, model_path, input_path):

    # Define the image size for the model input
    imageSize = (224, 224)
    # Determine whether to use GPU or CPU for inference
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define a list of labels for the output classes
    label_name = ['Gryllotalpa', 'Pieris rapae Linnaeus', 'Spodoptera litura', 'locust', 'stinkbug']

    # 1. Read the input image
    image = Image.open(input_path)
    # 2. Resize the image to the required input size of the model
    image = image.resize(imageSize)
    # 3. Load the trained model
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
         model.eval()
    # 4. Convert the image to a tensor
    transform = transforms.ToTensor()
    x = transform(image)
    x = torch.unsqueeze(x, 0)  # Add a dimension to match the batch size of the model
    x = x.to(device)
    # 5. Feed the tensor to the model for inference
    output = torch.squeeze(model(x))
    # 6. Get the class with the highest probability as the prediction
    output_class = torch.argmax(output)
    # Get the probability value of the predicted class
    output = torch.softmax(output, dim=0)
    print("Result: ", label_name[output_class], "Probability: ", output[output_class].item())
    return label_name[output_class], output[output_class].item()


if __name__ == '__main__':
    # Define the model architecture
    model = models.resnet50()
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),  # Activation functions
        nn.Dropout(0.4),  # Dropout is to prevent overfitting
        nn.Linear(256, 5),
        nn.LogSoftmax(dim=1)
    )
    # Make a prediction using the trained model and the input image
    pre(model, "./best_model_resnet50.pth", "C:/Users/20786/PycharmProjects/bg/dataset/test/Gryllotalpa/101id_Gryllotalpa.jpeg")





