import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerActivation
import numpy as np
import faiss

def get_embeddings(model, layer_num, input_data):
    # Get the specific layer
    layer = dict(model.named_children())[layer_num]
    # Let's say we have a model and a specific layer we're interested in
    layer_activations = LayerActivation(model, layer)
    # Calculate the activations (embeddings)
    embeddings = layer_activations.attribute(input_data)
    return embeddings

def calculate_cosine_similarity(model, layer_num, input_data1, input_data2):
    embedding1 = get_embeddings(model, layer_num, input_data1)
    embedding2 = get_embeddings(model, layer_num, input_data2)
    x = embedding1.detach().numpy()
    q = embedding2.detach().numpy()
    x = x.reshape(-1, np.prod(x.shape))
    q = q.reshape(-1, np.prod(q.shape))
    d = x.shape[1]  # Number of columns in x
    index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.ntotal
    faiss.normalize_L2(x)
    index.add(x)
    faiss.normalize_L2(q)
    distance, index = index.search(q, 5)
    print('Distance by FAISS:{}'.format(distance))

    #To Tally the results check the cosine similarity of the following example

    from scipy import spatial

    result = 1 - spatial.distance.cosine(embedding1.flatten(), embedding2.flatten())
    print('Distance by scipy:{}'.format(result))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
input1 = torch.randn(1, 1, 32, 32)
input2 = torch.randn(1, 1, 32, 32)
calculate_cosine_similarity(net, 'conv2', input1, input2)