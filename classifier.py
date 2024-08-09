from bag_of_words import train_vectors, train_labels, test_vectors, test_labels
from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Train a Decision Tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(train_vectors, train_labels)

# Evaluate the classifier
decision_tree_train_predictions = classifier.predict(
    train_vectors)
decision_tree_test_predictions = classifier.predict(test_vectors)

decision_tree_test_accuracy = accuracy_score(
    test_labels, decision_tree_test_predictions)
decision_tree_recall_tree = recall_score(
    test_labels, decision_tree_test_predictions, average='macro')


print("Decision Tree Accuracy:", decision_tree_test_accuracy)
print("Decision Tree Recall:", decision_tree_recall_tree)