import tarfile
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split


# Extract the data from tar.gz file
def extract_tar_gz(archive_path, extract_path):
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)


# Load data from extracted directories
# 819 .txt files for "atheism" category
# 1000 .txt files for "christianity" category
def load_data(directory, test_size, random_state):
    texts = []
    labels = []
    label_names = ['atheism', 'christianity']

    # label = 0 for atheism, 1 for christianity
    for label, sub_dir in enumerate(label_names):
        sub_dir_path = os.path.join(directory, sub_dir)
        for filename in os.listdir(sub_dir_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(sub_dir_path, filename)
                for encoding in ['utf-8', 'latin1', 'utf-16']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            texts.append(file.read())
                        break
                    except (UnicodeDecodeError, FileNotFoundError, PermissionError):
                        continue
                labels.append(label)

    # Split data into training and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state)

    # get the number of total data points and make sure the split is correct
    data_length = int(len(texts))
    assert len(train_texts) + len(test_texts) == data_length
    assert len(train_labels) == round(data_length*0.8)
    assert len(test_labels) == round(data_length*0.2)

    return train_texts, train_labels, test_texts, test_labels, label_names


# Define the data path and the path to extract the data
original_data_path = './data/religion_dataset.tar.gz'
extract_to_path = './data/extracted'
extract_tar_gz(original_data_path, extract_to_path)

# Update this with the path to your data directory
extracted_data_dir = './data/extracted/'
train_data, train_labels, test_data, test_labels, class_names = load_data(
    extracted_data_dir, test_size=0.2, random_state=42)

print("Data loaded successfully!")