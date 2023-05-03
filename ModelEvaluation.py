# Import necessary libraries
from itertools import cycle
import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import interp
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score

# Define the model evaluation function
def model_evaluation(model, model_path, data, save_path):
    # Determine whether the model should be run on GPU or CPU
    model = model.to('cuda')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path))
    label_name = ['Gryllotalpa', 'Pieris rapae Linnaeus', 'Spodoptera litura', 'locust', 'stinkbug']
    confusion = ConfusionMatrix(len(label_name), label_name, save_path)

    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()

    y_true = []
    y_pred = []
    y_score = []

    # Use the corresponding data set for testing, and record the data needed to evaluate the model
    for j, (inputs, labels) in enumerate(data):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        score_tmp = outputs
        _, preds = torch.max(outputs,1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_score.extend(score_tmp.detach().cpu().numpy())

    # Relevant metrics for model evaluation are calculated
    confusion.update(y_pred, y_true)
    confusion.summary()
    confusion.plot()
    confusion.roc(y_score, y_true)

    # Calculating accuracy
    acc = accuracy_score(y_true, y_pred)
    print('Accuracy: {:.4f}'.format(acc))

    # Calculating Precision
    precision = precision_score(y_true, y_pred, average='macro')
    print('Precision: {:.4f}'.format(precision))

    # Calculating recall
    recall = recall_score(y_true, y_pred, average='macro')
    print('Recall: {:.4f}'.format(recall))

    # Calculating F1_score
    f1 = f1_score(y_true, y_pred, average='macro')
    print('F1 Score: {:.4f}'.format(f1))

# Define the confusion matrix class
class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list, save_path):
        # Initialize the confusion matrix with 0 entries
        self.matrix = np.zeros((num_classes, num_classes))
        # Number of classes
        self.num_classes = num_classes
        # Class labels
        self.labels = labels
        # Path for saving the confusion matrix
        self.save_path = save_path
        # Accuracy of the model
        self.acc = 0

    # Calculate the confusion matrix
    def update(self, preds, labels):
        self.matrix=confusion_matrix(labels, preds)

    # Define the summary function to calculate index functions
    def summary(self):
        # Calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # The sum of the diagonal elements of the confusion matrix is the number of correct classifications
        self.acc = sum_TP / n
        print("the model accuracy is ", self.acc)

        # Calculate kappa
        sum_po = 0
        sum_pe = 0

        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    # Confusion matrix visualization
    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # Set the X-axis label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # Set the label Y-axis coordinate
        plt.yticks(range(self.num_classes), self.labels)
        # Display colorbar
        plt.colorbar()
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion matrix (acc=' + str(self.acc) + ')')
        plt.savefig(self.save_path + "Confusion matrix.jpg")

        #Set the threshold value and change the corresponding color according to the quantity
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

    # Define the ROC function
    def roc(self, y_score, y_test):
        # Set the number of classes
        num_class = 5

        # Convert y_score to numpy array
        score_array = np.array(y_score)

        # Convert y_test to one-hot format
        label_tensor = torch.tensor(y_test)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], num_class)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)

        # Calculate the fpr and tpr for each class using sklearn package
        fpr_dict = dict()
        tpr_dict = dict()
        roc_auc_dict = dict()
        for i in range(num_class):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        # Calculate micro-average fpr and tpr
        fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
        roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

        # Calculate macro-average fpr and tpr
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_class):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= num_class
        fpr_dict["macro"] = all_fpr
        tpr_dict["macro"] = mean_tpr
        roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

        # Plot the ROC curves for all classes
        plt.figure()
        lw = 2
        plt.plot(fpr_dict["micro"], tpr_dict["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc_dict["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr_dict["macro"], tpr_dict["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc_dict["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'yellow'])
        label_name = ['Gryllotalpa', 'Pieris rapae Linnaeus', 'Spodoptera litura', 'locust', 'stinkbug']
        for i, color in zip(range(num_class), colors):
            plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                     label='ROC curve of class ' + label_name[i] + ' (area = {0:0.4f})'
                                                                   ''.format(roc_auc_dict[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig(self.save_path + 'roc_curve.jpg')
        plt.show()

        #Start preparing the pr curve
        precision_dict = dict()
        recall_dict = dict()
        average_precision_dict = dict()
        for i in range(num_class):
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(label_onehot[:, i], score_array[:, i])
            average_precision_dict[i] = average_precision_score(label_onehot[:, i], score_array[:, i])

        # micro
        precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(label_onehot.ravel(),
                                                                                  score_array.ravel())
        average_precision_dict["micro"] = average_precision_score(label_onehot, score_array, average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(
            average_precision_dict["micro"]))

        # Plot the pr curve averaged over all categories
        plt.figure()
        plt.step(recall_dict['micro'], precision_dict['micro'], where='post')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision_dict["micro"]))
        plt.savefig(self.save_path+"pr_curve.jpg")
        plt.show()