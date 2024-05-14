import matplotlib.pyplot as plt
import numpy as np

# Function to parse the data from the file
def parse_data(file_path):
    scores = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Extracting the model name and the scores
            model_name = line.split(' [')[0]
            if "generated" not in model_name: ################### TEMP SHITTY CODE!
                continue
            scores_str = line.split(' [')[1].rstrip().rstrip(']\n')
            scores_list = [float(score) if score != 'nan' else None for score in scores_str.split(', ')]
            scores[model_name] = scores_list
    return scores


def assign_color(model_name, color_map):
    # base_name = model_name.split('_')[0]
    base_name=model_name ################### TEMP SHITTY CODE!
    if base_name not in color_map:
        color_map[base_name] = "C{}".format(len(color_map))  # Assign a new color
    return color_map[base_name]

# Path to your file
file_path = './model_ct_scores.txt'  

# Parsing the file contents
model_scores = parse_data(file_path)

# Rearrange the model_scores into a 2D numpy array
model_names = list(model_scores.keys())
scores = list(model_scores.values())

training_sizes = []
for model_name in model_names:
    if "full" in model_name:
        size = 50000
        training_sizes.append(size)
    else:
        size = int(model_name.split('_')[3])
        training_sizes.append(size)

zipped = zip(training_sizes, scores)
zipped = sorted(zipped, key=lambda x: x[0])
training_sizes, scores = zip(*zipped)

    
num_models = len(training_sizes)
num_scores = len(scores[0])
score_matrix = np.zeros((num_models, num_scores))
for i, score in enumerate(scores):
    score_matrix[i, :] = score


# Ranking
rankings = (-score_matrix).argsort(axis=0,).argsort(axis=0) + 1
average_rankings = rankings.mean(axis=1)
print(f"Average Rankings: {average_rankings}")
print(f"Training Sizes: {training_sizes}")





# Plotting the Average Ranking w.r.t. the training set size
plt.figure(figsize=(12, 6))
plt.plot(training_sizes, average_rankings, marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('Average Ranking')
plt.title('Average Ranking vs Training Set Size')
plt.grid(True)
plt.tight_layout()

# Loop through the data points to annotate each point
for i, (x, y) in enumerate(zip(training_sizes, average_rankings)):
    plt.annotate(f'({x}, {y:.1f})',  # text to display
                 (x, y),        # the poi·nt to annotate
                 textcoords="offset points",  # how to position the text
                 xytext=(0,10),  # distance from text to points (x,y)
                 ha='center')    # horizontal alignment can be left, right or center



plt.savefig('ct_ranking_plot_DDIM.png')