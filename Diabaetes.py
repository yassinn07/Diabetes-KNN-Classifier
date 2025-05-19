import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

diabetes_data = pd.read_csv('diabetes.csv')
print("\n\n")
print("KNN Model \n\n")

def euclidean_distance(instance1, instance2):
    return np.sqrt(np.sum((instance1 - instance2) ** 2))


def knn_classifier(train_data, train_labels, test_instance, k):
    distances = [(euclidean_distance(train_instance, test_instance), train_label)
                 for train_instance, train_label in zip(train_data, train_labels)]

    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]

    class_votes = {}
    for distance, label in k_nearest_neighbors:
        if label in class_votes:
            class_votes[label] += 1 / distance
        else:
            class_votes[label] = 1 / distance

    predicted_class = max(class_votes, key=class_votes.get)
    return predicted_class


def evaluate_accuracy(predictions, test_labels):
    correct_predictions = np.sum(predictions == test_labels)
    total_instances = len(test_labels)
    accuracy = correct_predictions / total_instances * 100
    return correct_predictions, total_instances, accuracy


k_values = [2, 3, 4]

all_accuracies = []

for k in k_values:
    accuracies = []

    train_data, test_data, train_labels, test_labels = train_test_split(
            diabetes_data.drop("Outcome", axis=1), diabetes_data["Outcome"], test_size=0.3, random_state=42
        )

    train_data_normalized = (train_data - train_data.min()) / (train_data.max() - train_data.min())
    test_data_normalized = (test_data - train_data.min()) / (train_data.max() - train_data.min())


    predictions = []
    for i in range(len(test_data_normalized)):
        test_instance = test_data_normalized.iloc[i, :]
        predicted_class = knn_classifier(train_data_normalized.values, train_labels.values, test_instance.values, k)
        predictions.append(predicted_class)

    correct, total, accuracy = evaluate_accuracy(np.array(predictions), test_labels.values)
    accuracies.append(accuracy)

    print(f'k value: {k}')
    print(f'Number of correctly classified instances: {correct}')
    print(f'Total number of instances: {total}')
    print(f'Accuracy: {accuracy:.2f}%')
    print('-------------------------')
    average_accuracy = np.mean(accuracies)
    all_accuracies.append(average_accuracy)
overall_average_accuracy = np.mean(all_accuracies)
print(f' Avg. Accuracy: {overall_average_accuracy:.2f}%')