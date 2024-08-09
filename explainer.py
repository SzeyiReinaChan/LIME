import random
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from data_handling import test_data, test_labels, class_names
from bag_of_words import bag_of_word
from classifier import classifier
from sklearn.linear_model import lars_path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


# This function perturbs the text by removing random words, this way we are able to
# get a new text that is similar to the original text but with some words removed.
# These new text are the local data set that we will use to train the model.
# text: the original text
# num_perturbations: number of perturbations we want to create
# Algo step: z' <- sample around x'
def perturb_text(text, num_perturbations):
    # First split the text into separate words
    words = text.split()
    perturbations = []
    for _ in range(num_perturbations):
        # create a copy of the word
        perturbed_text = words.copy()
        # randomly remove words from the text
        # at least remove one word
        range_ = random.randint(1, len(words))
        # print(range_)
        # remove the words & make sure the selection is random
        for _ in range(range_):
            # print("before", perturbed_text)
            # randomly select a word to remove, it can be any word in the text
            perturbed_random_index = random.randint(0, len(words) - 1)
            perturbed_text[perturbed_random_index] = ""
            # print(perturbed_text)
        perturbations.append(" ".join(perturbed_text))
    return perturbations
# # Test perturb_text()
# text = "This is a test sentence"
# num_perturbations = 5
# num_words = 2
# perturbations = perturb_text(text, num_perturbations)
# print(perturbations)


# This function computes the weights for the perturbations. (pi_x(z)))
# The weights are computed using the cosine distance between the original text and the perturbed text.
def compute_weights(original_vector, pert_vectors):
    weights = []
    for pert_vector in pert_vectors:
        distance = cosine_distances(original_vector, pert_vector)[0][0]
        weights.append(np.exp(-distance))
    return np.array(weights)

# # Test compute_weights()
# original_vector = np.array([[1, 0, 1, 0, 1]])
# pert_vectors = np.array([[1, 0, 0, 0, 1], [0, 0, 1, 0, 1]])
# weights = compute_weights(original_vector, pert_vectors)
# print(weights)


# This function predicts the probabilities of the perturbations using the trained classifier. (f(z))
def predict_probabilities(classifier, perturbations):
    # transform the perturbations text into number vectors using the bag of words
    perturbation_vectors = bag_of_word.transform(perturbations)
    # returns the probability of each class for the input samples
    return classifier.predict_proba(perturbation_vectors)


# Fit a sparse local linear model to the perturbed instances
def fit_local_model(perturbations, weights, predictions, num_features):
    # transform the perturbations into vectors
    perturbation_vectors = bag_of_word.transform(perturbations)
    # standardize the data
    scaler = StandardScaler(with_mean=False)
    # fit_transform() fits the data and then transforms it
    perturbation_vectors = scaler.fit_transform(perturbation_vectors).toarray()
    # compute LARS path
    # using the LARS algorithm to compute the Lasso path because it is efficient
    # when the number of features is large, the model runs faster compared to other methods
    _, _, coefs = lars_path(
        perturbation_vectors, predictions, method='lasso')
    # getting the coefficients
    coef = coefs[:, -1]
    # getting the feature names using the bag of words
    feature_names = bag_of_word.get_feature_names_out()
    # select top features based on absolute value of coefficients
    top_features = np.argsort(np.abs(coef))[-num_features:]
    return {feature_names[i]: coef[i] for i in top_features}


# Fit a model using random k features
def fit_random_k_features_model(perturbations, num_features):
    # Transform the perturbations into vectors
    pert_vectors_k_features = bag_of_word.transform(perturbations).toarray()

    # Randomly select k features
    feature_indices = np.random.choice(
        pert_vectors_k_features.shape[1], num_features, replace=False)

    # Get the selected feature names
    feature_names = bag_of_word.get_feature_names_out()
    selected_feature_names = [feature_names[i] for i in feature_indices]

    return selected_feature_names


# # Fit a greedy model by selecting features iteratively
# def fit_greedy_model(perturbations, weights, predictions, num_features):
#     pert_vectors_greedy = bag_of_word.transform(perturbations).toarray()
#     scaler = StandardScaler()
#     pert_vectors_greedy = scaler.fit_transform(pert_vectors_greedy)
#     feature_names = bag_of_word.get_feature_names_out()

#     # initialize the selected and remaining features
#     # remaining features are all the all available features after the selected features
#     selected_features = []
#     remaining_features = list(range(pert_vectors_greedy.shape[1]))
#     residuals = predictions.copy()

#     for _ in range(num_features):
#         best_feature = None
#         best_score = -np.inf
#         best_coef = None

#         for feature in remaining_features:
#             X_candidate = pert_vectors_greedy[:, feature].reshape(-1, 1)
#             model = LinearRegression()
#             model.fit(X_candidate, residuals, sample_weight=weights)
#             score = model.score(X_candidate, residuals, sample_weight=weights)

#             if score > best_score:
#                 best_score = score
#                 best_feature = feature
#                 best_coef = model.coef_[0]

#         if best_feature is not None:
#             selected_features.append(best_feature)
#             remaining_features.remove(best_feature)

#             # Update residuals
#             X_best = pert_vectors_greedy[:, best_feature].reshape(-1, 1)
#             model = LinearRegression()
#             model.fit(X_best, residuals, sample_weight=weights)
#             residuals -= model.predict(X_best) * best_coef

#     return {feature_names[i]: best_coef for i in selected_features}

# Function to run the explainer
def run_explainer(explainer, classifier, text, num_perturbations=1000, num_features=10):
    perturbations = perturb_text(text, num_perturbations)
    original_vector = bag_of_word.transform([text])
    pert_vectors = bag_of_word.transform(perturbations)
    weights = compute_weights(original_vector, pert_vectors)
    predictions = predict_probabilities(classifier, perturbations)
    if explainer == 'lime':
        explanation = fit_local_model(
            perturbations, weights, predictions[:, 1], num_features)
    elif explainer == 'random_k_features':
        explanation = fit_random_k_features_model(perturbations, num_features)
    # elif explainer == 'greedy':
    #     explanation = fit_greedy_model(text, classifier, perturbations)
    return explanation


# # total test instances = 364 instances
total_instances = len(test_data)
# Generate LIME explanations for each instance in the test set and write to a file
output_file = 'lime_explanations.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for i, text_instance in enumerate(test_data):
        print(f"Instance {i+1} out of {total_instances} in LIME")
        predicted_label = classifier.predict(
            bag_of_word.transform([text_instance]))[0]
        # Progress counter
        f.write(f"Instance {i+1} out of {total_instances}\n")
        # Print first 100 characters for short
        f.write(f"Text: {text_instance[:100]}...\n")
        f.write(f"Predicted class: {class_names[predicted_label]}\n")

        explanation = run_explainer('lime',
                                    classifier, text_instance, num_features=10)

        f.write("LIME Explanation:\n")
        for feature, weight in explanation.items():
            f.write(f"{feature}: {weight}\n")
        f.write("\n-----------------------------------\n")

    print(
        f"LIME explanations written to {output_file}")

# Generate Random K Features explanations for each instance in the test set and write to a file
output_file = 'random_k_features_explanations.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for i, (text_instance, true_label) in enumerate(zip(test_data, test_labels)):
        print(f"Instance {i+1} out of {total_instances} in Random K Features")
        predicted_label = classifier.predict(
            bag_of_word.transform([text_instance]))[0]
        f.write(f"Instance {i+1} out of {total_instances}\n")
        f.write(f"Text: {text_instance[:100]}...\n")
        f.write(
            f"Predicted class: {class_names[predicted_label]}, True class: {class_names[true_label]}\n")
        explanation = run_explainer('random_k_features',
                                    classifier, text_instance, num_features=10)
        f.write("Random K Features Explanation:\n")
        for feature in explanation:
            f.write(f"{feature}\n")
        f.write("\n-----------------------------------\n")

    print(f"Random K Features explanations written to {output_file}")


# # Generate Greedy explanations for each instance in the test set and write to a file
# output_file = 'greedy_explanations.txt'
# with open(output_file, 'w', encoding='utf-8') as f:
#     for i, (text_instance, true_label) in enumerate(zip(test_data, test_labels)):
#         print(f"Instance {i+1} out of {total_instances} in Greedy")
#         predicted_label = classifier.predict(
#             bag_of_word.transform([text_instance]))[0]
#         f.write(f"Instance {i+1} out of {total_instances}\n")
#         f.write(f"Text: {text_instance[:100]}...\n")
#         f.write(
#             f"Predicted class: {class_names[predicted_label]}, True class: {class_names[true_label]}\n")
#         explanation = run_explainer(
#             'greedy', classifier, text_instance, num_features=10)
#         f.write("Greedy Explanation:\n")
#         for feature, weight in explanation.items():
#             f.write(f"{feature}: {weight}\n")
#         f.write("\n-----------------------------------\n")
#     print(f"Greedy explanations written to {output_file}")


#   function to get the gold set of features from the decision tree
#   this is used to measure the recall of the explainer
def get_gold_features(model, num_features=10):
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[-num_features:]
    feature_names = bag_of_word.get_feature_names_out()
    return [feature_names[i] for i in top_indices]


# Function to measure recall of the explainer
def measure_explainer_recall(explainer, classifier, test_data, num_perturbations=1000, num_features=10):
    gold_features_set = []
    explanation_features_set = []

    for text in test_data:
        gold_features = get_gold_features(classifier, num_features)
        gold_features_set.append(gold_features)

        explanation = run_explainer(
            explainer, classifier, text, num_perturbations, num_features)
        if explainer == 'random_k_features':
            explanation_features = explanation
        else:
            explanation_features = list(explanation.keys())
        explanation_features_set.append(explanation_features)

    # Flatten the lists
    gold_features_flat = [
        item for sublist in gold_features_set for item in sublist]
    explanation_features_flat = [
        item for sublist in explanation_features_set for item in sublist]

    # Calculate recall
    recall = len(set(gold_features_flat) & set(
        explanation_features_flat)) / len(set(gold_features_flat))
    return recall

# Train a decision tree model with max depth of 10 and max features of 10 for the gold set
decision_tree_model = DecisionTreeClassifier(max_depth=10, max_features=10)
decision_tree_model.fit(bag_of_word.transform(test_data), test_labels)

# Measure recall for different explainers using the decision tree
recall_lime = measure_explainer_recall(
    'lime', decision_tree_model, test_data)
# recall_greedy = measure_explainer_recall(
#     'greedy', decision_tree_model, test_data)
recall_random_k = measure_explainer_recall(
    'random_k_features', decision_tree_model, test_data)

print(f"Recall of LIME: {recall_lime}")
# print(f"Recall of Greedy: {recall_greedy}")
print(f"Recall of Random K Features: {recall_random_k}")
