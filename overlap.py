import re

# this function reads the explanations from the file and returns a dictionary for overlapping features
def read_explanations(file_path, is_lime=False):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    explanations = {}
    instance_id = None
    capture_features = False
    for line in lines:
        line = line.strip()
        if line.startswith("Instance"):
            instance_id = int(line.split()[1])
            explanations[instance_id] = []
        elif line.startswith("Random K Features Explanation:") or line.startswith("LIME Explanation:"):
            capture_features = True
        elif capture_features and line and not line.startswith("-----------------------------------"):
            if is_lime:
                # Extract the feature name only, removing any numeric values (weights)
                # Remove numbers, colons, and whitespace
                feature_name = re.sub(r'[:\d\.\-\s]+$', '', line)
                explanations[instance_id].append(
                    feature_name.lower())  # Convert to lowercase
            else:
                explanations[instance_id].append(
                    line.lower())  # Convert to lowercase
        elif line.startswith("-----------------------------------"):
            capture_features = False

    return explanations


# Load explanations from both files
lime_explanations = read_explanations(
    'lime_explanations.txt', is_lime=True)
random_k_explanations = read_explanations(
    'random_k_features_explanations.txt')


overlap_scores = []
# Open a file to write the results
with open('feature_overlap_results.txt', 'w') as output_file:

    # Compare the explanations for each instance and calculate the overlap
    for instance_id in lime_explanations:
        lime_features = set(lime_explanations[instance_id])
        random_k_features = set(random_k_explanations.get(instance_id, []))

        # calculate intersection (common features)
        common_features = lime_features.intersection(random_k_features)
        # calculate overlap percentage
        if len(lime_features) > 0:
            overlap_percentage = (len(common_features) /
                                  len(lime_features)) * 100
        else:
            overlap_percentage = 0

        overlap_scores.append(overlap_percentage)
        output_file.write(f"Instance {instance_id}:\n")
        output_file.write(f"LIME Features: {', '.join(lime_features)}\n")
        output_file.write(
            f"Random K Features: {', '.join(random_k_features)}\n")
        output_file.write(f"Common Features: {', '.join(common_features)}\n")
        output_file.write(f"Overlap Percentage: {overlap_percentage:.2f}%\n")
        output_file.write("-----------------------------------\n\n")

    # Calculate the overall average overlap
    overall_overlap = sum(overlap_scores) / \
        len(overlap_scores) if overlap_scores else 0
    output_file.write(
        f"Overall Average Overlap Percentage: {overall_overlap:.2f}%\n")
