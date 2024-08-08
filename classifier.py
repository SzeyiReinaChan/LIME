from bag_of_words import train_vectors, train_labels, test_vectors, test_labels, unique_words
from data_handling import class_names
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Train a Decision Tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(train_vectors, train_labels)

# Evaluate the classifier
train_predictions = classifier.predict(train_vectors)
test_predictions = classifier.predict(test_vectors)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print("Training accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

tree_rules = export_text(classifier, feature_names=list(unique_words))
print(tree_rules)

# # Display the classifications for the training data
# print("\nTraining Data Classifications:")
# for text, label, prediction in zip(train_vectors, train_labels, train_predictions):
#     print(f"Text: {text[:60]}...")  # Print first 60 characters for shortness
#     print(
#         f"Actual: {class_names[label]}, Predicted: {class_names[prediction]}")
#     print("-" * 80)

# # Display the classifications for the test data
# print("\nTest Data Classifications:")
# for text, label, prediction in zip(test_vectors, test_labels, test_predictions):
#     print(f"Text: {text[:60]}...")  # Print first 60 characters for shortness
#     print(
#         f"Actual: {class_names[label]}, Predicted: {class_names[prediction]}")
#     print("-" * 80)
