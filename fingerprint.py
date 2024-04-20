import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_data(file_path):
    """Parse the model scores from a file."""
    scores = {}
    with open(file_path, 'r') as file:
        for line in file:
            model_name = line.split(' [')[0]
            scores_str = line.split(' [')[1].rstrip().rstrip(']\n')
            scores_list = [float(score) if score != 'nan' else None for score in scores_str.split(', ')]
            scores[model_name] = scores_list
    return scores

def assign_color(model_name, color_map):
    """Assign a color to each model for plotting."""
    base_name = model_name.split('_')[0]
    if base_name not in color_map:
        color_map[base_name] = "C{}".format(len(color_map))
    return color_map[base_name]

def plot_scores(file_path):
    """Plot the model scores."""
    model_scores = parse_data(file_path)
    plt.figure(figsize=(12, 6))
    color_map = {}
    for model, scores in model_scores.items():
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

def combined_metric(cosine_sim, l2_dist, alpha=0.5):
    """
    Combine cosine similarity and L2 distance into a single metric.
    Alpha determines the weight of the cosine similarity relative to the L2 distance.
    Cosine similarity is given more importance with a higher alpha.
    """
    # Normalize L2 distance to a similarity measure
    l2_similarity = 1 / (1 + l2_dist)
    # Combine the two metrics
    return alpha * cosine_sim + (1 - alpha) * l2_similarity


def find_closest_score_name(file_path, new_score, alpha=0.5):
    """Find the closest score name to a given score."""
    scores = parse_data(file_path)
    best_metric = -np.inf
    closest_score_name = None
    for model_name, model_scores in scores.items():
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

def compute_accuracy(file_path, alpha=1):
    """Compute the overall accuracy by setting each item as a target in turns."""
    scores = parse_data(file_path)
    correct_identifications = 0
    for target_name, target_scores in scores.items():
        best_metric = -np.inf
        closest_score_name = None
        for model_name, model_scores in scores.items():
            if model_name == target_name or None in model_scores:  # Skip self comparison and invalid scores
                continue
            model_scores = [0 if score is None else score for score in model_scores]
            target_scores_clean = [0 if score is None else score for score in target_scores]
            similarity = cosine_similarity(np.array(model_scores), np.array(target_scores_clean))
            l2_dist = l2_norm(np.array(model_scores), np.array(target_scores_clean))
            combined = combined_metric(similarity, l2_dist, alpha)
            if combined > best_metric:
                best_metric = combined
                closest_score_name = model_name
        if target_name.split('_')[0] == closest_score_name.split('_')[0]:
            correct_identifications += 1
        else:
            print(target_name, closest_score_name)
    accuracy = correct_identifications / len(scores)
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Process model scores.")
    parser.add_argument('file_path', type=str, help="Path to the score data file.")
    parser.add_argument('--plot', '-p', action='store_true', help="Plot the model scores.")
    parser.add_argument('--find_closest', '-f', action='store_true', help="Find the closest score name to a hardcoded list of scores.")
    parser.add_argument('--accuracy', '-a', action='store_true', help="Compute the overall accuracy.")
    args = parser.parse_args()

    if args.plot:
        plot_scores(args.file_path)
        print("Model scores plotted.")
    elif args.accuracy:
        accuracy = compute_accuracy(args.file_path)
        print(f"Overall accuracy: {accuracy}")
    elif args.find_closest is not None:
        # Hardcoded scores - modify this list as needed
        hardcoded_scores =  [-0.9282533426373575, 0.2091691723364699, 1.2879342619260798, 2.000266767399825, 2.0926155840977323, 2.0908728386222553, 1.8870787713504376, 2.3681898618924184, 2.583302407776081, 2.4658132270635016, 3.476056241779327, 3.443036285310274]
        closest_score_name = find_closest_score_name(args.file_path, hardcoded_scores)
        print(f"The closest score name is: {closest_score_name}")

if __name__ == "__main__":
    main()
