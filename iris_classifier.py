import cv2
import numpy as np
import csv

# Create lists for training/test data
train_features = []
train_labels = []
test_features = []
test_labels = []

# Load data from CSV file
with open('iris.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        if i < 140:
            train_features.append(row[:4])
            train_labels.append(np.eye(3)[int(row[4])])
        else:
            test_features.append(row[:4])
            test_labels.append(np.eye(3)[int(row[4])])

# Create neural network
net = cv2.ml.ANN_MLP_create()
net.setLayerSizes(np.array([4, 5, 5, 3]))
net.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
net.setTermCriteria((cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 1000, 0.001))

# Train neural network
net.train(np.array(train_features, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(train_labels, dtype=np.float32))

# Classify test points
pred = net.predict(np.array(test_features, dtype=np.float32))

# Print error
error = np.mean(np.square(np.array(test_labels, dtype=np.float32) - pred[1]))
print('Prediction error: {}'.format(error))
