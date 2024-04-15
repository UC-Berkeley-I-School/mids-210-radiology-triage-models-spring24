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
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from torch.utils.data import WeightedRandomSampler
import random
from torchvision import transforms
from torchvision.transforms import functional

########################################################################################################################
# Preprocess and augment images
########################################################################################################################

# Define the base transform
base_transform = transforms.Compose([
    transforms.Resize((256, 256)),
])


def resize_images(path):
    first_image_printed = False
    for image_name in os.listdir(path):
        if not image_name.endswith('.jpg'):
            continue  # Skip non-image files

        image_path = os.path.join(path, image_name)
        image = Image.open(image_path)
        image = base_transform(image)
        image.save(image_path)


def generate_csv_with_only_image_names(directories, csv_file_path):
    # Define the columns for the new CSV file
    columns = ['image_name']
    rows = []

    for directory_path in directories:
        for image_name in os.listdir(directory_path):
            if not image_name.endswith('.jpg'):
                continue  # Skip non-image files

            # Only add the image name to the row
            row = [image_name]
            rows.append(row)

    # Create a DataFrame with the collected rows and the specified column
    all_df = pd.DataFrame(rows, columns=columns)

    # Save the DataFrame to a CSV file
    all_df.to_csv(csv_file_path, index=False)


def generate_csv_with_set_images(directory, csv_path, num):
    # Define the columns for the new CSV file
    columns = ['image_name']
    rows = []

    for image_name in os.listdir(directory):
        if not image_name.endswith('.jpg'):
            continue  # Skip non-image files

            # Only add the image name to the row
        row = [image_name]
        rows.append(row)

    if num == 1:
        train_df = pd.DataFrame(rows, columns=columns)
        train_df.to_csv(csv_path, index=False)
    elif num == 2:
        test_df = pd.DataFrame(rows, columns=columns)
        test_df.to_csv(csv_path, index=False)
    else:
        val_df = pd.DataFrame(rows, columns=columns)
        val_df.to_csv(csv_path, index=False)


def update_set_with_labels(og_csv_path, final_csv_path, updated_csv_path):
    # Load the original dataset
    og_df = pd.read_csv(og_csv_path)

    # Load the final set that currently only contains image names
    final_df = pd.read_csv(final_csv_path)

    # Prepare additional columns for the final_df
    final_df['study_id'] = ''
    final_df['no_findings'] = 0
    final_df['atelectasis'] = 0
    final_df['cardiomegaly'] = 0
    final_df['lung_opacity'] = 0
    final_df['pleural_effusion'] = 0

    # Iterate through final_df to match and update details from og_df
    for index, row in final_df.iterrows():
        image_name = row['image_name']
        # Find the row in og_df that matches the image name
        matching_row = og_df[og_df.iloc[:, 10] == image_name]  # Assuming column 10 has image names

        if not matching_row.empty:
            # Directly assign the study_id
            final_df.at[index, 'study_id'] = matching_row.iloc[0, 1]  # Study ID from og.csv

            # Extract conditions and assign them individually
            final_df.at[index, 'atelectasis'] = matching_row.iloc[0, 2]  # Atelectasis
            final_df.at[index, 'cardiomegaly'] = matching_row.iloc[0, 3]  # Cardiomegaly
            final_df.at[index, 'lung_opacity'] = matching_row.iloc[0, 5]  # Lung Opacity
            final_df.at[index, 'pleural_effusion'] = matching_row.iloc[0, 6]  # Pleural Effusion

            # Compute 'no_findings' and assign
            conditions = matching_row.iloc[0, [2, 3, 5, 6]].values
            final_df.at[index, 'no_findings'] = int((conditions == 0).all())

    # Define the column order
    column_order = ['image_name', 'study_id', 'no_findings', 'atelectasis', 'cardiomegaly', 'lung_opacity',
                    'pleural_effusion']

    # Save the updated final set with the specified column order
    final_df.to_csv(updated_csv_path, columns=column_order, index=False)

    print("The updated CSV file has been saved.")


def augment_images_and_create_csv(image_directory, aug_directory, aug_csv_path):
    # Load the final CSV containing image names and their associated labels
    aug_df = pd.read_csv(aug_csv_path)

    # Ensure the augmented images directory exists
    os.makedirs(aug_directory, exist_ok=True)

    # Define the transformation: random rotation between 1 and 5 degrees
    rotate_transform = transforms.RandomRotation(degrees=(1, 5))

    for index, row in aug_df.iterrows():
        image_name = row['image_name']

        if image_name.endswith('.jpg'):
            # Construct the full image path
            image_path = os.path.join(image_directory, image_name)
            if not os.path.exists(image_path):
                continue  # If the image file does not exist, skip to the next row
            # Load the image
            image = Image.open(image_path)
            # Apply the random rotation
            rotated_image = rotate_transform(image)
            # Construct the new image name for the augmented image
            aug_image_name = f"r-{image_name}"
            # Construct the path for saving the augmented image
            aug_image_path = os.path.join(aug_directory, aug_image_name)
            # Save the augmented image
            rotated_image.save(aug_image_path)

            # replace current name with new image name
            aug_df.at[index, 'image_name'] = aug_image_name

    aug_df.to_csv(aug_csv_path, index=False)
    print("Augmentation complete, and aug.csv has been created.")


def append_csv_files(final_path, aug_path, full_train_path):
    # Load both CSV files into DataFrames
    base_df = pd.read_csv(final_path)
    new_df = pd.read_csv(aug_path)

    # Ensure column consistency (Optional: Based on your dataset's requirements)
    assert list(base_df.columns) == list(new_df.columns), "Columns do not match between CSV files."

    # Concatenate the new_df DataFrame to the end of base_df DataFrame
    combined_df = pd.concat([base_df, new_df], ignore_index=True)

    # Save the combined DataFrame back to the base_csv_path
    combined_df.to_csv(full_train_path, index=False)

    print(f"Contents of {aug_path} have been appended to {final_path}.\n")


def print_num_entries(img_csv, file_name):
    df_all = pd.read_csv(img_csv)
    num_entries = len(df_all)
    print(f"The number of entries in the {file_name} file is: {num_entries}\n")


og_csv_path = 'Processed_Image_Data_March_11_2024.csv'

test = 'mimic_images_March_10_2024_Datasets/prep/test'
train = 'mimic_images_March_10_2024_Datasets/prep/train'
val = 'mimic_images_March_10_2024_Datasets/prep/val'

image_csv_path = 'mimic_images_March_10_2024_Datasets/prep/image.csv'
train_csv_path = 'mimic_images_March_10_2024_Datasets/prep/train_image.csv'
test_csv_path = 'mimic_images_March_10_2024_Datasets/prep/test_image.csv'
val_csv_path = 'mimic_images_March_10_2024_Datasets/prep/val_image.csv'

train_label_path = 'mimic_images_March_10_2024_Datasets/prep/train_set_labeled.csv'
test_label_path = 'mimic_images_March_10_2024_Datasets/prep/test_set_labeled.csv'
val_label_path = 'mimic_images_March_10_2024_Datasets/prep/val_set_labeled.csv'

aug = 'mimic_images_March_10_2024_Datasets/prep/aug'
aug_label_path = 'mimic_images_March_10_2024_Datasets/prep/aug_img.csv'
full_train_path = 'mimic_images_March_10_2024_Datasets/prep/full_train_path.csv'

# Resize images
# resize_images(test)
# resize_images(train)
# resize_images(val)
print("Image resizing done")

# Generate the CSV file for all the images
# generate_csv_with_only_image_names([train, test, val], image_csv_path)
print_num_entries(image_csv_path, "ALL images")

# create CSV file with images, study_id and labels for test
# generate_csv_with_set_images(test, test_csv_path, 2)
# update_set_with_labels(og_csv_path, test_csv_path, test_label_path)
print_num_entries(test_label_path, "TEST images")

# create CSV file with images, study_id and labels for val
# generate_csv_with_set_images(val, val_csv_path, 3)
# update_set_with_labels(og_csv_path, val_csv_path, val_label_path)
print_num_entries(val_label_path, "VAL images")

# FIXME: Uncomment to run and train with unbalanced dataset
# # create CSV file with images, study_id and labels for train
# generate_csv_with_set_images(train, train_csv_path, 1)
# update_set_with_labels(og_csv_path, train_csv_path, train_label_path)
# print_num_entries(train_label_path, "TRAIN images")
#
# # augment and save images. Creat scv file containing labels
# augment_images_and_create_csv(train, aug, aug_label_path)
# print_num_entries(aug_label_path, "AUG images")
#
# # append augmented data to og data
# append_csv_files(train_label_path, aug_label_path, full_train_path)
# print_num_entries(full_train_path, "Full train images")

# FIXME: uncomment to run and train with balanced dataset
# Same as above but for balanced dataset
train_bal = 'mimic_images_March_10_2024_Datasets/prep/train_4_bal_s'
train_bal_path = 'mimic_images_March_10_2024_Datasets/prep/train_bal_image.csv'
train_bal_label_path = 'mimic_images_March_10_2024_Datasets/prep/train_bal_labeled.csv'

# resize_images(train_bal)

# create CSV file with images, study_id and labels for train_bal
# generate_csv_with_set_images(train_bal, train_bal_path, 1)
# update_set_with_labels(og_csv_path, train_bal_path, train_bal_label_path)
print_num_entries(train_bal_label_path, "TRAIN balanced images")

# augment and save images. Creat scv file containing labels
# augment_images_and_create_csv(train_bal, aug, aug_label_path)
print_num_entries(aug_label_path, "AUG images")

# append augmented data to og data
# append_csv_files(train_bal_label_path, aug_label_path, full_train_path)
print_num_entries(full_train_path, "Full train images")

########################################################################################################################
# create a dataset object and attaches labels to images
########################################################################################################################

loader_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MedicalImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.root_dir, str(img_id))
        try:
            image = Image.open(img_path)
        except FileNotFoundError:
            print(f"File {img_path} not used in this dataset.")
            return None
        print(f"File {img_path} found.")


        label = self.annotations.iloc[index, [2, 3, 4, 5, 6]].values.astype(np.int8)
        label = torch.tensor(label, dtype=torch.int8)

        if self.transform:
            image = self.transform(image)

        return img_id, image, label


def calculate_metrics(y_true, y_pred):
    y_pred = torch.sigmoid(y_pred).cpu().numpy()
    y_pred = np.round(y_pred)
    y_true = y_true.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, f1

if os.path.exists('train_loader.pkl'):
    # Load the DataLoader objects from the pickle files
    with open('train_loader.pkl', 'rb') as f:
        train_loader = pickle.load(f)
else:
    train_dataset = MedicalImageDataset(csv_file='mimic_images_March_10_2024_Datasets/prep/full_train_path.csv',
                                        root_dir='mimic_images_March_10_2024_Datasets/prep/all_train',
                                        transform=loader_transform)

    print('Removing None types for train dataset, this will take a long time.')
    indices = [1, 2, 3, 4]
    train_dataset = [data for data in train_dataset if data is not None]
    no_findings_labels = np.array([1 if sum([sample[2][i] for i in indices]) == 1 else 0 for sample in train_dataset])

    # Calculate weights: more weight to 'no_findings' == 1 samples
    weights = np.ones_like(no_findings_labels)
    weights[no_findings_labels == 1] = 2  # Adjust this weight as necessary

    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, len(weights))

    print('Finished removing None types for train dataset')

    # FIXME: uncomment for unbalanced
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, shuffle=False)

    # FIXME: uncomment for balanced
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    with open('train_loader.pkl', 'wb') as f:
        pickle.dump(train_loader, f)

if os.path.exists('test_loader.pkl'):
    with open('test_loader.pkl', 'rb') as f:
        test_loader = pickle.load(f)
else:
    test_dataset = MedicalImageDataset(csv_file='mimic_images_March_10_2024_Datasets/prep/test_set_labeled.csv',
                                       root_dir='mimic_images_March_10_2024_Datasets/prep/test',
                                       transform=loader_transform)

    # test_dataset = [data for data in test_dataset if data is not None]

    filtered_test_dataset = []

    for data in test_dataset:
        if data is not None:
            filtered_test_dataset.append(data)

    test_dataset = filtered_test_dataset

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    with open('test_loader.pkl', 'wb') as f:
        pickle.dump(test_loader, f)

if os.path.exists('validation_loader.pkl'):
    with open('validation_loader.pkl', 'rb') as f:
        validation_loader = pickle.load(f)

else:

    validation_dataset = MedicalImageDataset(csv_file='mimic_images_March_10_2024_Datasets/prep/val_set_labeled.csv',
                                             root_dir='mimic_images_March_10_2024_Datasets/prep/val',
                                             transform=loader_transform)

    print('Removing None types for train dataset, this will take a long time.')
    indices = [1, 2, 3, 4]
    validation_dataset = [data for data in validation_dataset if data is not None]
    no_findings_labels = np.array([1 if sum([sample[2][i] for i in indices]) == 1 else 0 for sample in validation_dataset])

    # Calculate weights: more weight to 'no_findings' == 1 samples
    val_weights = np.ones_like(no_findings_labels)
    val_weights[no_findings_labels == 1] = 5  # Adjust this weight as necessary

    # Create a WeightedRandomSampler
    val_sampler = WeightedRandomSampler(val_weights, len(val_weights))

    # validation_dataset = [data for data in validation_dataset if data is not None]

    validation_loader = DataLoader(validation_dataset, sampler=val_sampler, batch_size=32, shuffle=False)

    # Save the DataLoader objects
    with open('validation_loader.pkl', 'wb') as f:
        pickle.dump(validation_loader, f)

if os.path.exists('train_bal_loader.pkl'):
    with open('train_bal_loader.pkl', 'rb') as f:
        train_bal_loader = pickle.load(f)

else:

    train_bal_dataset = MedicalImageDataset(csv_file='mimic_images_March_10_2024_Datasets/prep/train_bal_labeled.csv',
                                             root_dir='mimic_images_March_10_2024_Datasets/prep/train_4_bal_s',
                                             transform=loader_transform)

    train_bal_dataset = [data for data in train_bal_dataset if data is not None]

    train_bal_loader = DataLoader(train_bal_dataset, batch_size=32, shuffle=False)

    # Save the DataLoader objects
    with open('train_bal_loader.pkl', 'wb') as f:
        pickle.dump(train_bal_loader, f)


iterator = iter(train_loader)
_, images, labels = next(iterator)
print(f'Image shape: {images.shape}')
print(f'Label shape: {labels.shape}')

for i, (_, images, labels) in enumerate(train_loader):
    if torch.cuda.is_available():
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

print(images.shape, labels.shape)


# Assuming train_loader, test_loader, and validation_loader are your DataLoader instances
# Assuming train_loader, test_loader, and validation_loader are your DataLoader instances

num_train_items = len(train_loader.dataset)
num_test_items = len(test_loader.dataset)
num_validation_items = len(validation_loader.dataset)
num_train_bal_items = len(train_bal_loader.dataset)

print(f"Number of items in training dataset: {num_train_items}")
print(f"Number of items in test dataset: {num_test_items}")
print(f"Number of items in validation dataset: {num_validation_items}")
print(f"Number of items in train_bal dataset: {num_train_bal_items}")

num_train_batches = len(train_loader)
num_test_batches = len(test_loader)
num_validation_batches = len(validation_loader)
num_train_bal_batches = len(train_bal_loader)

print(f"Number of batches in training DataLoader: {num_train_batches}")
print(f"Number of batches in test DataLoader: {num_test_batches}")
print(f"Number of batches in validation DataLoader: {num_validation_batches}")
print(f"Number of batches in train_bal DataLoader: {num_train_bal_batches}")





########################################################################################################################
# visualize data
########################################################################################################################

def get_all_labels(data_loader):  # Assuming you're passing a DataLoader, not Dataset
    all_labels_list = []
    for _, _, labels in data_loader:  # Adjusted to unpack correctly
        all_labels_list.append(labels.cpu().numpy())
    return np.concatenate(all_labels_list, axis=0)


class_names_list = ['no_findings', 'atelectasis', 'cardiomegaly', 'lung_opacity', 'pleural_effusion']

single_label_count = 0
multi_label_count = 0


# Go through the DataLoader and check labels
for _, _, labels in train_loader:
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


print(f"Single-label instances: {single_label_count}")
print(f"Multi-label instances: {multi_label_count}")

# Extract labels for each dataset
train_all_labels = get_all_labels(train_loader)
test_all_labels = get_all_labels(test_loader)
validation_all_labels = get_all_labels(validation_loader)

# Concatenate all labels across train, test, and validation sets
all_labels = np.concatenate([train_all_labels, test_all_labels, validation_all_labels], axis=0)

# Sum across samples to get the total count for each class
class_counts = np.sum(all_labels, axis=0)


# plt.figure(figsize=(10, 7))
sns.barplot(x=class_names_list, y=class_counts)
plt.title('Class Distribution Across Train, Test, Validation Sets')
plt.xlabel('Class')
plt.ylabel('Frequency')

# Corrected rotation without explicitly setting the labels
plt.xticks(rotation=45)

plt.show()

# Assuming train_loader is defined and is a DataLoader object
train_all_labels = get_all_labels(train_loader)

# Sum across samples to get the total count for each class in the training set
class_counts = np.sum(train_all_labels, axis=0)

# Plotting
sns.barplot(x=class_names_list, y=class_counts)
plt.title('Class Distribution in Training Set')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

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

    for img_ids, images, labels in train_loader:  # Updated to include img_ids
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

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
        for _, images, labels in validation_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
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
        for _, images, labels in validation_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5
            all_val_labels.append(labels.cpu())
            all_val_preds.append(predictions.cpu())

    # Concatenate all the predictions and labels
    all_val_labels = torch.cat(all_val_labels, dim=0)
    all_val_preds = torch.cat(all_val_preds, dim=0)

    # Calculate precision, recall, f1-score, and support
    target_names = ['no_findings', 'atelectasis', 'cardiomegaly', 'lung_opacity', 'pleural_effusion']
    report = classification_report(all_val_labels.numpy(), all_val_preds.numpy(), target_names=target_names)
    print(f'\nEpoch {epoch + 1}/{num_epochs} Classification Report:\n {report}')

    print(
        f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, '
        f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.4f}')

    # Step the scheduler
    scheduler.step()

# FIXME: Edit to save under the correct model
torch.save(model, 'EfficientNet-B3-Bal.pth')

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

# Binarize the labels for each class
y_bin = label_binarize(all_val_labels.numpy(), classes=[0, 1, 2, 3, 4])
n_classes = y_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# the predicted probabilities for each class
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_val_preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {i}')

plt.plot([0, 1], [0, 1], 'k--', label='No skill line')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for multi-class')
plt.legend(loc="lower right")
plt.show()

true_labels = all_val_labels.numpy()
print("train duplicate ", )

true_labels = np.argmax(true_labels, axis=1)

predicted_labels = np.argmax(all_val_preds.numpy(), axis=1)

cm = confusion_matrix(true_labels, predicted_labels)

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)

# Labels, title and ticks
label_names = ['No_Findings','Atelectasis', 'Cardiomegaly', 'Lung_Opacity', 'Pleural_Effusion']
ax.set_xlabel('Predicted labels', fontsize=18)
ax.set_ylabel('True labels', fontsize=18)
ax.set_title('Confusion Matrix', fontsize=18)
ax.xaxis.set_ticklabels(label_names, fontsize=12, rotation=45)
ax.yaxis.set_ticklabels(label_names, fontsize=12, rotation=0)
plt.show()

########################################################################################################################
# evaluate and collect
########################################################################################################################
def evaluate_and_collect_data(model, data_loader):
    model.eval()
    all_ids = []
    all_probs = []
    with torch.no_grad():
        for data in data_loader:
            if data is None:  # Skip the loop iteration if the dataset returned None
                continue
            img_ids, images, labels = data  # Now unpacking three values
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probabilities)
            all_ids.extend(list(img_ids))  # Make sure img_ids is iterable
    return all_ids, all_probs

def save_results_to_csv(ids, probs, file_name):
    # Assuming probs is a list of numpy arrays with the shape (num_samples, num_classes)
    df = pd.DataFrame(data=probs, columns=['No_Findings', 'Atelectasis', 'Cardiomegaly', 'Lung_Opacity', 'Pleural_Effusion'])
    df.insert(0, 'img_id', ids)
    df.to_csv(file_name, index=False)

# For validation set
val_ids, val_probs = evaluate_and_collect_data(model, validation_loader)
save_results_to_csv(val_ids, val_probs, 'validation_results.csv')

# For testing set
val_ids, val_probs = evaluate_and_collect_data(model, test_loader)
save_results_to_csv(val_ids, val_probs, 'test_results.csv')

# For training set
val_ids, val_probs = evaluate_and_collect_data(model, train_loader)
save_results_to_csv(val_ids, val_probs, 'train_results.csv')

# For balanced training set
val_ids, val_probs = evaluate_and_collect_data(model, train_bal_loader)
save_results_to_csv(val_ids, val_probs, 'train_bal_results.csv')

model.eval()  # Set the model to evaluation mode

dummy_input = torch.randn(1, 3, 256, 256, device='cuda')

# Specify the name of the ONNX file
output_onnx_file = 'EfficientNet-B3_Model_balanced.onnx'

# Export the model to an ONNX file
torch.onnx.export(model,
                  dummy_input,
                  output_onnx_file,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})


train_path = 'train_results.csv'
csv_file_train = pd.read_csv(train_path)
test_path = 'test_results.csv'
csv_file_test = pd.read_csv(test_path)
val_path = 'validation_results.csv'
csv_file_val = pd.read_csv(val_path)
train_bal_path = 'train_bal_results.csv'
csv_file_val = pd.read_csv(val_path)

test_duplicates = csv_file_test[csv_file_test.iloc[:, 0].duplicated(keep=False)]
val_duplicates = csv_file_val[csv_file_val.iloc[:, 0].duplicated(keep=False)]
train_duplicates = csv_file_train[csv_file_train.iloc[:, 0].duplicated(keep=False)]

if not test_duplicates.empty:
    print(f"Found {len(test_duplicates)} duplicates in test set:")
    print(test_duplicates)
else:
    print("No duplicates found in test set.")

if not val_duplicates.empty:
    print(f"Found {len(val_duplicates)} duplicates in val set:")
    print(val_duplicates)
else:
    print("No duplicates found in val set.")

if not train_duplicates.empty:
    print(f"Found {len(train_duplicates)} duplicates in train set:")
    print(train_duplicates)
else:
    print("No duplicates found in train set.")