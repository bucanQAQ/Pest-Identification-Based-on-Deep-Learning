import matplotlib.pyplot as plt
import numpy as np

# Define model names and evaluation metrics
model_names = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
save_path = "./models/no_rembg_models/"

#Define scores of evaluation metrics for each model

scores = np.array([[0.9809, 0.9813, 0.9815, 0.9814],
                   [0.9816, 0.9842, 0.9811, 0.9824],
                   [0.9770, 0.9758, 0.9791, 0.9772],
                   [0.9824, 0.9832, 0.9841, 0.9836],
                   [0.9900, 0.9899, 0.9909, 0.9904]])

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

scores = np.array([[0.980882, 0.0612],
                   [0.981618, 0.0440],
                   [0.976961, 0.0641],
                   [0.982353, 0.0514],
                   [0.989951, 0.0345]])

# Draw multiple bar charts
fig, ax1 = plt.subplots()

bar_width = 0.35
opacity = 0.8
index = np.arange(len(model_names))

# Draw an accuracy histogram for each model
rects1 = ax1.bar(index, scores[:,0], bar_width, alpha=opacity, color='b', label=metrics2[0])

# Create the second coordinate axis
ax2 = ax1.twinx()

# Draw a histogram of loss values for each model
rects2 = ax2.bar(index + bar_width, scores[:,1], bar_width, alpha=opacity, color='r', label=metrics2[1])
# for i in range(len(model_names)):
#     ax1.text(rects1[i].get_x() + rects1[i].get_width() / 2.0, scores[i, 0], str(scores[i, 0]), ha='center', va='bottom')
#     ax2.text(rects2[i].get_x() + rects2[i].get_width() / 2.0, scores[i, 1], str(scores[i, 1]), ha='center', va='bottom')

ax1.set_xlabel('Model')
ax1.set_ylabel(metrics2[0])
ax2.set_ylabel(metrics2[1])
ax1.set_ylim(0.95,1)
plt.xticks(index + bar_width/2, model_names)
plt.legend(handles=[rects1, rects2], loc='lower right')


plt.tight_layout()
plt.title("Accuracy and Loss (no remove background)")
plt.show()

plt.savefig(save_path + "Accuracy and Loss (remove background).jpeg")