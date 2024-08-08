import random
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from data_handling import test_data, test_labels, class_names
from bag_of_words import bag_of_word
from classifier import classifier
from sklearn.linear_model import lars_path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


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
    perturbation_vectors = bag_of_word.transform(perturbations)
    scaler = StandardScaler(with_mean=False)
    perturbation_vectors = scaler.fit_transform(perturbation_vectors).toarray()
    # Compute LARS path
    _, _, coefs = lars_path(
        perturbation_vectors, predictions, method='lasso')
    # Select the coefficients for a particular alpha (e.g., the smallest one)
    coef = coefs[:, -1]  # Get coefficients for the smallest alpha
    feature_names = bag_of_word.get_feature_names_out()
    # Select top features based on absolute value of coefficients
    top_features = np.argsort(np.abs(coef))[-num_features:]
    return {feature_names[i]: coef[i] for i in top_features}


def lime_text_explainer(classifier, text, num_perturbations=1000, num_features=10):
    perturbations = perturb_text(text, num_perturbations)
    original_vector = bag_of_word.transform([text])
    pert_vectors = bag_of_word.transform(perturbations)
    weights = compute_weights(original_vector, pert_vectors)
    predictions = predict_probabilities(classifier, perturbations)
    # Use probabilities for the positive class
    explanation = fit_local_model(
        perturbations, weights, predictions[:, 1], num_features)
    return explanation


def fit_random_k_features_model(perturbations, weights, predictions, num_features):
    pert_vectors_k_features = bag_of_word.transform(perturbations).toarray()
    feature_indices = np.random.choice(
        pert_vectors_k_features.shape[1], num_features, replace=False)
    selected_features = pert_vectors_k_features[:, feature_indices]

    scaler = StandardScaler()
    selected_features = scaler.fit_transform(selected_features)

    model = LinearRegression()
    model.fit(selected_features, predictions, sample_weight=weights)

    feature_names = bag_of_word.get_feature_names_out()
    selected_feature_names = [feature_names[i] for i in feature_indices]

    return {selected_feature_names[i]: model.coef_[i] for i in range(num_features)}


def random_k_features_explainer(classifier, text, num_perturbations=1000, num_features=10):
    perturbations = perturb_text(text, num_perturbations)
    original_vector = bag_of_word.transform([text])
    pert_vectors = bag_of_word.transform(perturbations)
    weights = compute_weights(original_vector, pert_vectors)
    predictions = predict_probabilities(classifier, perturbations)
    explanation = fit_random_k_features_model(
        perturbations, weights, predictions[:, 1], num_features)
    return explanation


# Generate LIME explanations for each instance in the test set and write to a file
# total test instances = 364 instances
output_file = 'lime_explanations.txt'
total_instances = len(test_data)
with open(output_file, 'w', encoding='utf-8') as f:
    for i, text_instance in enumerate(test_data):
        predicted_label = classifier.predict(
            bag_of_word.transform([text_instance]))[0]
        # Progress counter
        f.write(f"Instance {i+1} out of {total_instances}\n")
        # Print first 100 characters for brevity
        f.write(f"Text: {text_instance[:100]}...\n")
        f.write(f"Predicted class: {class_names[predicted_label]}\n")

        explanation = lime_text_explainer(
            classifier, text_instance, num_features=10)

        f.write("LIME Explanation:\n")
        for feature, weight in explanation.items():
            f.write(f"{feature}: {weight}\n")
        f.write("\n-----------------------------------\n")

    print(
        f"LIME explanations written to {output_file}")
    
# Generate Random K Features explanations for each instance in the test set and write to a file
output_file = 'random_k_features_explanations.txt'
total_instances = len(test_data)
with open(output_file, 'w', encoding='utf-8') as f:
    for i, (text_instance, true_label) in enumerate(zip(test_data, test_labels)):
        predicted_label = classifier.predict(
            bag_of_word.transform([text_instance]))[0]
        f.write(f"Instance {i+1} out of {total_instances}\n")
        f.write(f"Text: {text_instance[:100]}...\n")
        f.write(
            f"Predicted class: {class_names[predicted_label]}, True class: {class_names[true_label]}\n")
        explanation = random_k_features_explainer(
            classifier, text_instance, num_features=10)
        f.write("Random K Features Explanation:\n")
        for feature, weight in explanation.items():
            f.write(f"{feature}: {weight}\n")
        f.write("\n-----------------------------------\n")
        
    print(f"Random K Features explanations written to {output_file}")
