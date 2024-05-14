import matplotlib.pyplot as plt

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

# Plotting
plt.figure(figsize=(12, 6))
color_map = {}
for i in range(len(training_sizes)):
    model = f"{training_sizes[i]} Images"
    color = assign_color(model, color_map)
    plt.plot(scores[i], label=model, color=color)






# # Plotting
# plt.figure(figsize=(12, 6))
# color_map = {}
# for model, scores in model_scores.items():
#     if "full" in model:
#         model = "50000 Images"
#     else:
#         size = model.split('_')[3]
#         model = f"{size} Images"

#     color = assign_color(model, color_map)
#     plt.plot(scores, label=model, color=color)

plt.xlabel('Score Index')
plt.ylabel('Score Value')
plt.title('Model Scores')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.grid(True)
plt.tight_layout()

plt.savefig('ct_plots_DDIM.png')
