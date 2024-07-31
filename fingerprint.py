import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_data(file_path):
    """Parse the model scores from a file."""
    train_scores = {}
    test_scores = {}
    with open(file_path, 'r') as file:
        for line in file:
            model_name = line.split(' [')[0]
            scores_str = line.split(' [')[1].rstrip().rstrip(']\n')
            scores_list = [float(score) if score != 'nan' else None for score in scores_str.split(', ')]
            if '_train' in model_name:
                if model_name not in train_scores:
                    train_scores[model_name] = []
                train_scores[model_name].append(scores_list)
            elif '_test' in model_name:
                test_scores[model_name] = scores_list
    return train_scores, test_scores

def assign_color(model_name, color_map):
    """Assign a color to each model for plotting."""
    base_name = model_name.split('_')[0]
    if base_name not in color_map:
        color_map[base_name] = "C{}".format(len(color_map))
    return color_map[base_name]

def plot_scores(file_path):
    """Plot the model scores."""
    train_scores, test_scores = parse_data(file_path)
    plt.figure(figsize=(12, 6))
    color_map = {}
    for model, scores in {**train_scores, **test_scores}.items():
        color = assign_color(model, color_map)
        plt.plot(scores, label=model, color=color)
    plt.xlabel('Score Index')
    plt.ylabel('Score Value')
    plt.title('Model Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_scores_plot.png')

def cosine_similarity(A, B):
    """Compute the cosine similarity between two vectors."""
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)

def l2_norm(A, B):
    """Compute the L2 norm (Euclidean distance) between two vectors."""
    return np.linalg.norm(A - B)

def combined_metric(cosine_sim, l2_dist, alpha):
    """
    Combine cosine similarity and L2 distance into a single metric.
    Alpha determines the weight of the cosine similarity relative to the L2 distance.
    Cosine similarity is given more importance with a higher alpha.
    """
    # Normalize L2 distance to a similarity measure
    l2_similarity = 1 / (1 + l2_dist)
    # Combine the two metrics
    return alpha * cosine_sim + (1 - alpha) * l2_similarity


def find_closest_score_name(train_scores, new_score, alpha):
    """Find the closest score name to a given score."""
    best_metric = -np.inf
    closest_score_name = None
    for model_name, model_scores in train_scores.items():
        if None in model_scores:
            model_scores = [0 if score is None else score for score in model_scores]
        if len(model_scores) != len(new_score):
            continue
        similarity = cosine_similarity(np.array(model_scores), np.array(new_score))
        l2_dist = l2_norm(np.array(model_scores), np.array(new_score))
        combined = combined_metric(similarity, l2_dist, alpha)
        if combined > best_metric:
            best_metric = combined
            closest_score_name = model_name
    return closest_score_name

def compute_accuracy(file_path, alpha=0.5):
    """Compute the overall accuracy by setting each test item as a target and finding the best match in train data."""
    train_scores, test_scores = parse_data(file_path)
    correct_identifications = 0
    for test_name, test_scores_list in test_scores.items():
        best_metric = -np.inf
        closest_score_name = None
        for train_name, train_scores_list in train_scores.items():
            for score_list in train_scores_list:
                train_scores_list = [0 if score is None else score for score in score_list][1:]
                test_scores_clean = [0 if score is None else score for score in test_scores_list][1:]
                similarity = cosine_similarity(np.array(train_scores_list), np.array(test_scores_clean))
                l2_dist = l2_norm(np.array(train_scores_list), np.array(test_scores_clean))
                combined = combined_metric(similarity, l2_dist, alpha)
                if combined > best_metric:
                    best_metric = combined
                    closest_score_name = train_name
        if test_name.split('_')[0] == closest_score_name.split('_')[0]:
            correct_identifications += 1
        else:
            print(f"Test: {test_name}, Closest Train: {closest_score_name}")
    accuracy = correct_identifications / len(test_scores)
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Process model scores.")
    parser.add_argument('file_path', type=str, help="Path to the score data file.")
    parser.add_argument('--plot', '-p', action='store_true', help="Plot the model scores.")
    parser.add_argument('--accuracy', '-a', action='store_true', help="Compute the overall accuracy.")
    args = parser.parse_args()

    if args.plot:
        plot_scores(args.file_path)
        print("Model scores plotted.")
    elif args.accuracy:
        accuracy = compute_accuracy(args.file_path)
        print(f"Overall accuracy: {accuracy}")

if __name__ == "__main__":
    main()