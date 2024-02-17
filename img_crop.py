import torch
import random
import os
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, load_from_disk
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import numpy as np
from IPython import display
import requests
from io import BytesIO
from PIL import Image
from PIL import Image, ImageSequence
from IPython.display import HTML
import warnings
from tqdm import tqdm
from matplotlib import rc
import gc
import matplotlib
import argparse
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
gc.enable()
plt.ioff()

parser = argparse.ArgumentParser(description="Model Training/Loading Script")
parser.add_argument("--train", help="Train a new model", action="store_true")
parser.add_argument("--load", help="Load an existing model", action="store_true")
args = parser.parse_args()

dataset = load_dataset("cifar10")
save_dir = "./processed_cifar10"

num_classes = 10
resnet = resnet18(pretrained=True)
resnet.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1)
resnet_ = list(resnet.children())[:-2]
resnet_[3] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
classifier = nn.Conv2d(512,num_classes,1)
torch.nn.init.kaiming_normal_(classifier.weight)
resnet_.append(classifier)
resnet_.append(nn.Upsample(size=32, mode='bilinear', align_corners=False))
tiny_resnet = nn.Sequential(*resnet_)

def attention(x):
    return torch.sigmoid(torch.logsumexp(x,1, keepdim=True))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = CIFAR10(root='.', train=True, download=True, transform=transform_train)
train_iter = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

testset = CIFAR10(root='.', train=False, download=True, transform=transform_test)
test_iter = DataLoader(testset, batch_size=100, shuffle=False, num_workers=16, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = nn.DataParallel(tiny_resnet).cuda()
num_epochs = 30
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,78,eta_min=0.001)


if args.train:
    losses = []
    acces = []
    v_losses = []
    v_acces = []
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        acc = 0.0
        var = 0.0
        model.train()
        train_pbar = train_iter
        for i, (x, _label) in enumerate(train_pbar):
            x = x.cuda()
            _label = _label.cuda()
            label = F.one_hot(_label).float()
            seg_out = model(x)
            
            attn = attention(seg_out)
            # Smooth Max Aggregation
            logit = torch.log(torch.exp(seg_out*0.5).mean((-2,-1)))*2
            loss = criterion(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            acc += (logit.argmax(-1)==_label).sum()
            #train_pbar.set_description('Accuracy: {:.3f}%'.format(100*(logit.argmax(-1)==_label).float().mean()))
            
        avg_loss = epoch_loss / (i + 1)
        losses.append(avg_loss)
        avg_acc = acc.cpu().detach().numpy() / (len(trainset))
        acces.append(avg_acc)
        model.eval()
        epoch_loss = 0.0
        acc = 0.0
        num_seen = 0
        
        test_pbar = tqdm(test_iter)
        for i, (x, _label) in enumerate(test_pbar):
            x = x.cuda()
            _label = _label.cuda()
            label = F.one_hot(_label).float()
            seg_out = model(x)
            attn = attention(seg_out)
            logit = torch.log(torch.exp(seg_out*0.5).mean((-2,-1)))*2
            loss = criterion(logit, label)
            epoch_loss += loss.item()
            acc += (logit.argmax(-1)==_label).sum()
            num_seen += label.size(0)
            test_pbar.set_description('Accuracy: {:.3f}%'.format(100*(acc.float()/num_seen)))
        
        avg_loss_val = epoch_loss / (i + 1)
        v_losses.append(avg_loss_val)
        avg_acc_val = acc.cpu().detach().numpy() / (len(testset))
        v_acces.append(avg_acc_val)
        plt.close('all')

        conf = torch.max(nn.functional.softmax(seg_out, dim=1), dim=1)[0]
        hue = (torch.argmax(seg_out, dim=1).float() + 0.5)/10
        x -= x.min()
        x /= x.max()
        gs_im = x.mean(1)
        gs_mean = gs_im.mean()
        gs_min = gs_im.min()
        gs_max = torch.max((gs_im-gs_min))
        gs_im = (gs_im - gs_min)/gs_max
        hsv_im = torch.stack((hue.float(), attn.squeeze().float(), gs_im.float()), -1)
        im = hsv_to_rgb(hsv_im.cpu().detach().numpy())
        ex = make_grid(torch.tensor(im).permute(0,3,1,2), normalize=True, nrow=25)
        attns = make_grid(attn, normalize=False, nrow=25)
        attns = attns.cpu().detach()
        inputs = make_grid(x, normalize=True, nrow=25).cpu().detach()
        display.clear_output(wait=True)
        plt.figure(figsize=(20,8))
        plt.imshow(np.concatenate((inputs.numpy().transpose(1,2,0),ex.numpy().transpose(1,2,0), attns.numpy().transpose(1,2,0)), axis=0))
        #plt.xticks(np.linspace(18,324,10), classes)
        #plt.xticks(fontsize=20) 
        plt.yticks([])
        plt.title('CIFAR10 Epoch:{:02d}, Train:{:.3f}, Test:{:.3f}'.format(epoch, avg_acc, avg_acc_val), fontsize=20)
        plt.gcf()
        plt.savefig("myplot.jpg")
        fig, ax = plt.subplots(1,2, figsize=(20,8))
        ax[0].set_title('Crossentropy')
        ax[0].plot(losses, label='Train')
        ax[0].plot(v_losses, label='CIFAR10 Test')
        ax[0].legend()
        ax[1].set_title('Accuracy')
        ax[1].plot(acces, label='Train')
        ax[1].plot(v_acces, label='CIFAR10 Test')
        ax[1].legend()
        display.display(plt.gcf())  

        torch.save(model.state_dict(), 'tiny_resnet_model.pth')
elif args.load:
    model.load_state_dict(torch.load('tiny_resnet_model.pth'))
    model.eval()  # Set the model to evaluation mode
else:
    print("No action specified. Please use --train or --load.")

def create_gaussian_noise_background(image_size, mean=127, std=100):
    """Create a Gaussian noise background"""
    gaussian_noise = np.random.normal(mean, std, image_size + (3,))
    gaussian_noise = np.clip(gaussian_noise, 0, 255).astype(np.uint8)
    return Image.fromarray(gaussian_noise)

def get_random_bg20k_image(bg20k_dataset_dir, size):
    # List all files in the BG-20k dataset directory
    bg_images = [f for f in os.listdir(bg20k_dataset_dir) if os.path.isfile(os.path.join(bg20k_dataset_dir, f))]
    
    # Randomly select an image file
    random_file = random.choice(bg_images)
    random_image_path = os.path.join(bg20k_dataset_dir, random_file)
    random_image = Image.open(random_image_path)

    # Resize the image
    return random_image.resize(size)

def crop_main_body_noisy(element):
    # Preprocess the image and forward through the model
    image = element['img']
    label = element['label']
    input_tensor = transform_test(image).unsqueeze(0).cuda()
    with torch.no_grad():
        seg_out = model(input_tensor)

    attn = attention(seg_out)   
    attn = attn.squeeze().cpu().numpy()
    threshold = 0.5  # Define a threshold value
    mask = attn > threshold

    # Convert the mask to a PIL image
    mask = Image.fromarray(mask.astype(np.uint8) * 255).convert('L')
    background = create_gaussian_noise_background(image.size)
    image_rgba = image.convert('RGBA')
    cropped_image = Image.composite(image_rgba, background, mask)

    save_image(np.array(cropped_image), os.path.join(save_dir, 'segmented_realworld'), label)
    element['img'] = np.array(cropped_image)
    return element

def crop_main_body_realworld(element):
    # Preprocess the image and forward through the model
    image = element['img']
    label = element['label']
    input_tensor = transform_test(image).unsqueeze(0).cuda()
    with torch.no_grad():
        seg_out = model(input_tensor)

    attn = attention(seg_out)   
    attn = attn.squeeze().cpu().numpy()
    threshold = 0.5  # Define a threshold value
    mask = attn > threshold

    # Convert the mask to a PIL image
    mask = Image.fromarray(mask.astype(np.uint8) * 255).convert('L')
    background = get_random_bg20k_image("/home/jackhe/background/testval",image.size)
    image_rgba = image.convert('RGBA')
    cropped_image = Image.composite(image_rgba, background, mask)

    save_image(np.array(cropped_image), os.path.join(save_dir, 'segmented_realworld'), label)
    element['img'] = np.array(cropped_image)
    return element

name_count = 0
def save_image(img_array, base_save_path, class_label):
    global name_count
    class_folder = os.path.join(base_save_path, str(class_label))
    os.makedirs(class_folder, exist_ok=True)  # Create the class folder if it doesn't exist
    img = Image.fromarray(img_array)
    img.save(os.path.join(class_folder, f"{name_count}.png"))
    name_count+=1

seg_dataset = dataset['train'].map(crop_main_body_realworld)
seg_dataset.save_to_disk("./processed_cifar10/seg_dataset_realworld")


