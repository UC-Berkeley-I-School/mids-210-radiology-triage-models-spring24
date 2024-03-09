########################################################################################################################
# Radiology Capstone
# Created by Cinthya Rosales
########################################################################################################################

from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
import os
import numpy as np
import pickle
from torchvision.models.resnet import ResNet18_Weights
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import timm
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F

########################################################################################################################
# Reduce the number of no_findings in dataset
########################################################################################################################

# # Load dataset
# csv_file = pd.read_csv('Processed_Image_Data_Feb_28_2024.csv')
# df = '/home/ai/PycharmProjects/radiologycastone/pythonProject2/df'
#
# # Initialize the counter and the list for removal
# no_findings_counter = 0
# max_no_findings = 2050
# remove_list = []
#
# # Iterate through the DataFrame
# for index, row in csv_file.iterrows():
#     items = []
#     for i in range(2,8):
#         t = int(row[i])
#         items.append(t)
#     if sum(items) == 0.0:  # Adjust indexes if needed
#         no_findings_counter += 1
#         # If the counter exceeds the maximum allowed 'no findings', add to remove list
#         if no_findings_counter > max_no_findings:
#             temp = (row[10])
#             remove_list.append(temp)
#
# print('Remove list size: ', len(remove_list))
#
# # Iterate over the list of images to drop
# for image_name in remove_list:
#     # Construct the full path to the image file
#     image_path = os.path.join(df, image_name + '.jpg')
#
#     # Check if the file exists before attempting to delete it
#     if os.path.exists(image_path):
#         os.remove(image_path)
#         print(f"Removed {image_path}")
#     else:
#         print(f"File {image_path} not found")

########################################################################################################################
# Re-split the dataset into test, training validation. 70,15,15 split
########################################################################################################################

# df = '/home/ai/PycharmProjects/radiologycastone/pythonProject2/df'
# base_dir = '/home/ai/PycharmProjects/radiologycastone/pythonProject2'
#
# # Create subdirectories for the train, test, and validation splits
# train_dir = os.path.join(base_dir, 'train')
# test_dir = os.path.join(base_dir, 'test')
# val_dir = os.path.join(base_dir, 'validation')
#
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)
# os.makedirs(val_dir, exist_ok=True)
#
# # List all images in the original folder
# images = [img for img in os.listdir(df) if img.endswith('.jpg')]
#
# # Split the images list into training (70%) and a temporary subset (30%)
# train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
#
# # Split the temporary subset into test and validation sets (50% each of the 30%)
# test_images, val_images = train_test_split(temp_images, test_size=0.5, random_state=42)
#
# def copy_images(image_list, source_dir, target_dir):
#     for image_name in image_list:
#         source_path = os.path.join(source_dir, image_name)
#         destination_path = os.path.join(target_dir, image_name)
#         shutil.copy(source_path, destination_path)
#
# # Copy images to their respective directories
# copy_images(train_images, df, train_dir)
# copy_images(test_images, df, test_dir)
# copy_images(val_images, df, val_dir)
#
# # Optionally, print the number of images in each directory
# print(f"Training set size: {len(os.listdir(train_dir))}")
# print(f"Testing set size: {len(os.listdir(test_dir))}")
# print(f"Validation set size: {len(os.listdir(val_dir))}")

########################################################################################################################
# create a dataset object and attaches labels to images
########################################################################################################################

class MedicalImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 10]
        img_path = os.path.join(self.root_dir, str(img_id) + '.jpg')
        try:
            image = Image.open(img_path)
        except FileNotFoundError:
            print(f"File {img_path} not used in this dataset.")
            return None
        print(f"File {img_path} found.")

        label = self.annotations.iloc[index, [2, 3, 5, 6]].values.astype(np.int8)
        # Check if all the entries for pathological labels are 0, which means no findings
        if sum(label) == 0:
            no_findings = int(np.all(label == 0))
            # Add the no_findings label to the label array
            label = np.append(label, no_findings)
            label = torch.tensor(label, dtype=torch.int8)
        else:
            label = np.append(label, 0)
            label = torch.tensor(label, dtype=torch.int8)

        if self.transform:
            image = self.transform(image)

        return (image, label)

def calculate_metrics(y_true, y_pred):
    y_pred = torch.sigmoid(y_pred).cpu().numpy()
    y_pred = np.round(y_pred)
    y_true = y_true.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, f1


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-18 expects images of size 224x224
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if os.path.exists('train_loader.pkl'):
    # Load the DataLoader objects from the pickle files
    with open('train_loader.pkl', 'rb') as f:
        train_loader = pickle.load(f)
else:
    train_dataset = MedicalImageDataset(csv_file='Processed_Image_Data_Feb_28_2024.csv',
                                        root_dir='train',
                                        transform=transform)

    print('Removing None types for train dataset, this will take a long time.')
    train_dataset = [data for data in train_dataset if data is not None]
    print('Finished removing None types for train dataset')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    with open('train_loader.pkl', 'wb') as f:
        pickle.dump(train_loader, f)

if os.path.exists('test_loader.pkl'):
    with open('test_loader.pkl', 'rb') as f:
        test_loader = pickle.load(f)
else:
    test_dataset = MedicalImageDataset(csv_file='Processed_Image_Data_Feb_28_2024.csv',
                                       root_dir='test',
                                       transform=transform)

    print('Removing None types for test dataset')
    test_dataset = [data for data in test_dataset if data is not None]
    print('Finished removing None types for test dataset')

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    with open('test_loader.pkl', 'wb') as f:
        pickle.dump(test_loader, f)

if os.path.exists('validation_loader.pkl'):
    with open('validation_loader.pkl', 'rb') as f:
        validation_loader = pickle.load(f)

else:

    validation_dataset = MedicalImageDataset(csv_file='Processed_Image_Data_Feb_28_2024.csv',
                                             root_dir='validation',
                                             transform=transform)

    print('Removing None types for validation dataset')
    validation_dataset = [data for data in validation_dataset if data is not None]
    print('Finished removing None types for validation dataset')

    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

    # Save the DataLoader objects
    with open('validation_loader.pkl', 'wb') as f:
        pickle.dump(validation_loader, f)
iterator = iter(train_loader)
images, labels = next(iterator)
print(f'Image shape: {images.shape}')
print(f'Label shape: {labels.shape}')

for i, (images, labels) in enumerate(train_loader):
    if torch.cuda.is_available():
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

print(images.shape, labels.shape)


########################################################################################################################
# visualize data
########################################################################################################################

def get_all_labels(dataloader):
    all_labels_list = []
    for _, labels in dataloader:
        # Convert labels to CPU and to NumPy, then append to list
        all_labels_list.append(labels.cpu().numpy())
    # Concatenate list of arrays into a single array
    return np.concatenate(all_labels_list, axis=0)


class_names_list = ['atelectasis', 'cardiomegaly', 'lung_opacity', 'pleural_effusion', 'no_findings']

single_label_count = 0
multi_label_count = 0

# Placeholder to store the count of each category
category_counts = {}

# Go through the DataLoader and check labels
for _, labels in train_loader:
    # Assuming your labels are torch Tensors
    numpy_labels = labels.numpy()

    # Check each label in the batch
    for label in numpy_labels:
        # Count the number of pathologies in each label
        pathology_count = np.sum(label)

        if pathology_count == 1:
            single_label_count += 1
        elif pathology_count > 1:
            multi_label_count += 1

        # Add to category count
        if pathology_count in category_counts:
            category_counts[pathology_count] += 1
        else:
            category_counts[pathology_count] = 1

print(f"Single-label instances: {single_label_count}")
print(f"Multi-label instances: {multi_label_count}")
print(f"Category counts: {category_counts}")


# Extract labels for each dataset
train_all_labels = get_all_labels(train_loader)
test_all_labels = get_all_labels(test_loader)
validation_all_labels = get_all_labels(validation_loader)

# Concatenate all labels across train, test, and validation sets
all_labels = np.concatenate([train_all_labels, test_all_labels, validation_all_labels], axis=0)

# Sum across samples to get the total count for each class
class_counts = np.sum(all_labels, axis=0)


# # plt.figure(figsize=(10, 7))
# sns.barplot(x=class_names_list, y=class_counts)
# plt.title('Class Distribution Across Train, Test, Validation Sets')
# plt.xlabel('Class')
# plt.ylabel('Frequency')
#
# # Corrected rotation without explicitly setting the labels
# plt.xticks(rotation=45)
#
# plt.show()

########################################################################################################################
# defines the Focal loss
########################################################################################################################
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduce=None)
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduce=None)

        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


########################################################################################################################
# defines class weights
#######################################################################################################################

count_atelectasis = np.sum(all_labels[:, 0])
count_cardiomegaly = np.sum(all_labels[:, 1])
count_lung_opacity = np.sum(all_labels[:, 2])
count_pleural_effusion = np.sum(all_labels[:, 3])
count_no_findings = np.sum(all_labels[:, 4])

class_samples = np.array([count_atelectasis, count_cardiomegaly, count_lung_opacity,
                          count_pleural_effusion, count_no_findings])
total_samples = class_samples.sum()
number_of_classes = len(class_samples)
class_weights = total_samples / (class_samples * number_of_classes)

class_weights_tensor = torch.FloatTensor(class_weights).cuda() if torch.cuda.is_available() \
    else torch.FloatTensor(class_weights)

########################################################################################################################
# Uncomment to choose custom CNN
########################################################################################################################

# class DeepCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(DeepCNN, self).__init__()
#         self.features = nn.Sequential(
#             # Conv Layer block 1
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),  # Batch Normalization after convolutions
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),  # Batch Normalization after convolutions
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             # Conv Layer block 2
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),  # Batch Normalization after convolutions
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),  # Batch Normalization after convolutions
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             # Conv Layer block 3
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),  # Batch Normalization after convolutions
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),  # Batch Normalization after convolutions
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#
#         self._to_linear = None
#
#         self._get_conv_output([1, 3, 224, 224])  # Example size [batch_size, channels, height, width]
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(self._to_linear, 1024),  # The first value is now dynamically assigned
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(512, num_classes),
#         )
#
#     # Method to calculate the output feature size of convolutions
#     def _get_conv_output(self, shape):
#         input = torch.autograd.Variable(torch.rand(shape))
#         output_feat = self.features(input)
#         self._to_linear = output_feat.data.view(1, -1).size(1)
#
#     def forward(self, x):
#         # Convolutional layers...
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         # Fully connected layers...
#         x = self.classifier(x)
#         return x
#
#
# # Assuming you are using 224x224 images, the input feature to the first FC layer needs to be adjusted.
# # If not, calculate the size of the feature maps after the last pooling layer and adjust accordingly.
#
# # Initialize the model
# model = DeepCNN(num_classes=5)
#
# # Initialize the optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
# criterion = FocalLoss(alpha=1, gamma=2, logits=True, reduce='mean')

########################################################################################################################
# Uncomment to choose ResNet-18
########################################################################################################################
#
# # Initialize ResNet-18 with pre-trained weights
# weights = ResNet18_Weights.IMAGENET1K_V1
# model = models.resnet18(weights=weights)

# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 7)  # Adjusting for 6 output classes

# # Unfreeze layer4 and layer3 parameters
# for param in model.layer4.parameters():
#     param.requires_grad = True
#
# for param in model.layer3.parameters():
#     param.requires_grad = True

# # Define your optimization and loss function here
# optimizer = optim.Adam([
#     {'params': model.layer3.parameters()},
#     {'params': model.layer4.parameters()},
#     {'params': model.fc.parameters()}
# ], lr=0.001, weight_decay=1e-5)

########################################################################################################################
# Uncomment to choose DenseNet-121
########################################################################################################################

# # Load a pre-trained DenseNet121
# model = models.densenet121(pretrained=True)
#
# num_features = model.classifier.in_features
# model.classifier = nn.Linear(num_features, 5)
#
# # Freeze all layers in the network
# for param in model.parameters():
#     param.requires_grad = False
#
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# criterion = nn.CrossEntropyLoss()

########################################################################################################################
# Uncomment to choose EfficientNet-B3
########################################################################################################################

# Initialize EfficientNet-b3 model with pre-trained weights
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=5)

# Modify the classifier for a 7-class problem
model.classifier = nn.Linear(model.classifier.in_features, 5)

# Freeze or unfreeze all the layers in the beginning
for param in model.parameters():
    param.requires_grad = True

# Uncomment and edit to determine # of layers to unfreeze
# #Specify the block number from which you want to start unfreezing
# unfreeze_from_block = 16
#
# # Unfreezing blocks from the specified block number to the last one
# for name, param in model.named_parameters():
#     # Unfreeze the specified blocks and the classifier
#     if name.startswith('blocks') and int(name.split('.')[1]) >= unfreeze_from_block:
#         param.requires_grad = True
#     elif name.startswith('classifier'):
#         param.requires_grad = True

# Define optimizer and loss function
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-5)

#criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
criterion = FocalLoss(alpha=1, gamma=2, logits=True, reduce='mean')
########################################################################################################################
# train model
########################################################################################################################

# Transfer the model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU!")

# # Training Loop
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Define scheduler
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds, total_preds = 0, 0

    for images, labels in train_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = torch.sigmoid(outputs) > 0.5
        correct_preds += (predictions == labels).float().sum()
        total_preds += torch.numel(predictions)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_preds / total_preds
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct_preds, val_total_preds = 0, 0
    with torch.no_grad():
        for images, labels in validation_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            val_running_loss += loss.item()
            predictions = torch.sigmoid(outputs) > 0.5
            val_correct_preds += (predictions == labels).float().sum()
            val_total_preds += torch.numel(predictions)

    val_epoch_loss = val_running_loss / len(validation_loader)
    val_epoch_acc = val_correct_preds / val_total_preds
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)

    all_val_labels = []
    all_val_preds = []
    with torch.no_grad():
        for images, labels in validation_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5
            all_val_labels.append(labels.cpu())
            all_val_preds.append(predictions.cpu())

    # Concatenate all the predictions and labels
    all_val_labels = torch.cat(all_val_labels, dim=0)
    all_val_preds = torch.cat(all_val_preds, dim=0)

    # Calculate precision, recall, f1-score, and support
    target_names = ['atelectasis', 'cardiomegaly', 'lung_opacity', 'pleural_effusion', 'no_findings']
    report = classification_report(all_val_labels.numpy(), all_val_preds.numpy(), target_names=target_names)
    print(f'\nEpoch {epoch + 1}/{num_epochs} Classification Report:\n {report}')

    print(
        f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, '
        f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.4f}')

    # Step the scheduler
    scheduler.step()

# FIXME: Edit to save under the correct model
torch.save(model, 'EfficientNet_B3.pth')

########################################################################################################################
# visualize results
########################################################################################################################

# Convert lists of tensors to NumPy arrays after moving them to CPU
train_losses_np = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
val_losses_np = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
train_accuracies_np = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in train_accuracies]
val_accuracies_np = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in val_accuracies]

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses_np, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses_np, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies_np, label='Training Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies_np, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

