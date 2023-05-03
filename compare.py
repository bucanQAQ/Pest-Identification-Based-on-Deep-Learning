import matplotlib.pyplot as plt
import numpy as np

# Define model names and evaluation metrics
model_names = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
save_path = "./models/rembg_models/"

#Define scores of evaluation metrics for each model
scores = np.array([[0.9700, 0.9692, 0.9709, 0.9699],
                   [0.9656, 0.9647, 0.9666, 0.9656],
                   [0.9664, 0.9650, 0.9674, 0.9660],
                   [0.9604, 0.9616, 0.9619, 0.9617],
                   [0.9651, 0.9664, 0.9652, 0.9657]])

#Create a bar chart showing evaluation metric scores for each model
fig, ax = plt.subplots()
bar_width = 0.2
opacity = 0.8
index = np.arange(len(model_names))


for i in range(len(metrics)):
    plt.bar(index + i*bar_width, scores[:,i], bar_width,
            alpha=opacity,
            label=metrics[i])

plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0.95,1)
plt.xticks(index + bar_width*2, model_names)
plt.legend(loc='lower right')
plt.title('Model Comparison(no remove background)')
plt.savefig(save_path + 'Model Comparison(remove background).jpeg')

plt.tight_layout()
plt.show()



#Define new evaluation metrics and scores for each model
metrics2 = ['Accuracy', 'Loss']


scores = np.array([[0.969986, 0.0949],
                   [0.965621, 0.0971],
                   [0.966439, 0.1071],
                   [0.960437, 0.1105],
                   [0.965075, 0.0948]])

#Create a bar chart showing evaluation metric scores for each model
fig, ax1 = plt.subplots()

bar_width = 0.35
opacity = 0.8
index = np.arange(len(model_names))


rects1 = ax1.bar(index, scores[:,0], bar_width, alpha=opacity, color='b', label=metrics2[0])


ax2 = ax1.twinx()


rects2 = ax2.bar(index + bar_width, scores[:,1], bar_width, alpha=opacity, color='r', label=metrics2[1])

ax1.set_xlabel('Model')
ax1.set_ylabel(metrics2[0])
ax2.set_ylabel(metrics2[1])
ax1.set_ylim(0.95,1)
ax2.set_ylim(0.09)
plt.xticks(index + bar_width/2, model_names)
plt.legend(handles=[rects1, rects2], loc='lower right')
plt.title("Accuracy and Loss(remove background)")

plt.tight_layout()
plt.show()

plt.savefig(save_path + "Accuracy and Loss(remove background).jpeg")
