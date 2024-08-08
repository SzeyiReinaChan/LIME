from data_handling import train_data, train_labels, test_data, test_labels, class_names
from sklearn.feature_extraction.text import CountVectorizer

# Initialize
bag_of_word = CountVectorizer()

# Fit the bag_of_word on the training data and transform the training data
train_vectors = bag_of_word.fit_transform(train_data)

# Transform the test data using the same bag_of_word
test_vectors = bag_of_word.transform(test_data)

# Get the unique words
unique_words = bag_of_word.get_feature_names_out()

# print(test_data[1])

# # vocabulary size
# print("Vocabulary size:", len(bag_of_word.vocabulary_))
# # train_vectors.shape & test_vectors.shape = (number of samples, vocabulary size)
# print("Training data shape:", train_vectors.shape)
# print("Test data shape:", test_vectors.shape)
# # # array of unique words
# # print("Feature names:", unique_words)

# print("Bag of words created successfully!")