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

def find_closest_score_name(file_path, new_score):
    """Find the closest score name to a given score."""
    scores = parse_data(file_path)
    highest_similarity = -1
    closest_score_name = None
    for model_name, model_scores in scores.items():
        if None in model_scores:
            model_scores = [0 if score is None else score for score in model_scores]
        if len(model_scores) != len(new_score):
            continue
        similarity = cosine_similarity(np.array(model_scores), np.array(new_score))
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_score_name = model_name
    return closest_score_name

def main():
    parser = argparse.ArgumentParser(description="Process model scores.")
    parser.add_argument('file_path', type=str, help="Path to the score data file.")
    parser.add_argument('--plot', '-p', action='store_true', help="Plot the model scores.")
    parser.add_argument('--find_closest', '-f', action='store_true', help="Find the closest score name to a hardcoded list of scores.")
    args = parser.parse_args()

    if args.plot:
        plot_scores(args.file_path)
        print("Model scores plotted.")
    elif args.find_closest is not None:
        # Hardcoded scores - modify this list as needed
        hardcoded_scores =  [-0.2150689843835414, -0.11082282357398737, 0.3520616382037417, 1.0121871907745663, 1.494801946221974, 1.703670869938864, 1.5354972338937725, 1.954209972721663, 1.9717661158133366, 1.9631870832166354, 3.3765989939633196, 4.745762368532049]
        closest_score_name = find_closest_score_name(args.file_path, hardcoded_scores)
        print(f"The closest score name is: {closest_score_name}")

if __name__ == "__main__":
    main()
