import matplotlib.pyplot as plt

# Function to parse the data from the file
def parse_data(file_path):
    scores = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Extracting the model name and the scores
            model_name = line.split(' [')[0]
            scores_str = line.split(' [')[1].rstrip().rstrip(']\n')
            scores_list = [float(score) if score != 'nan' else None for score in scores_str.split(', ')]
            scores[model_name] = scores_list
    return scores

def assign_color(model_name, color_map):
    base_name = model_name.split('_')[0]
    if base_name not in color_map:
        color_map[base_name] = "C{}".format(len(color_map))  # Assign a new color
    return color_map[base_name]

# Path to your file
file_path = './model_ct_scores.txt'  

# Parsing the file contents
model_scores = parse_data(file_path)

# Plotting
plt.figure(figsize=(12, 6))
color_map = {}
for model, scores in model_scores.items():
#    if 'Diffusion' in model:
#        continue
    color = assign_color(model, color_map)
    plt.plot(scores, label=model, color=color)

plt.xlabel('Score Index')
plt.ylabel('Score Value')
plt.title('Model Scores')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.grid(True)
plt.tight_layout()

plt.savefig('model_scores_plot.png')
