import os, sys
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
import transforms as T
import utils

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class ImagesDataset(torch.utils.data.Dataset):

    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width

        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [image for image in sorted(os.listdir(files_dir))
                        if image[-4:]=='.jpg']


        # classes: 0 index is reserved for background
        self.classes = ['background', 'missing', 'bite', 'open', 'short', 'spur', 'spurious']

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0

        # annotation file
        annot_filename = img_name[:-4] + '.xml'
        annot_file_path = os.path.join(  './PCB_DATASET/labels_all/' , annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]

        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            label_name = member.find('name').text
            if label_name == 'missing_hole':
                label_id = 1
            if label_name == 'mouse_bite':
                label_id = 2
            if label_name == 'open_circuit':
                label_id = 3
            if label_name == 'short':
                label_id = 4
            if label_name == 'spur':
                label_id = 5
            if label_name == 'spurious_copper':
                label_id = 6
            labels.append(label_id)

            # bounding box
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)

            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)


            xmin_corr = (xmin/wt)*self.width
            xmax_corr = (xmax/wt)*self.width
            ymin_corr = (ymin/ht)*self.height
            ymax_corr = (ymax/ht)*self.height

            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        if self.transforms:

            sample = self.transforms(image = img_res,
                                     bboxes = target['boxes'],
                                     labels = labels)

            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])



        return img_res, target

    def __len__(self):
        return len(self.imgs)

list_of_classes = ['background', 'missing', 'bite', 'open', 'short', 'spur', 'spurious']
def plot_img_bbox(img, target, img2, target2, scores):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,2)
    fig.set_size_inches(5,5)
    a[0].imshow(img)
    predicted_labels = target['labels']
    counter = 0
    for box in (target['boxes']):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a[0].add_patch(rect)
        label_id = predicted_labels[counter]
        a[0].text(x,(y-10),str(list_of_classes[label_id]), verticalalignment='top', color='white',fontsize=10,weight='bold')
        counter += 1

    a[1].imshow(img2)
    predicted_labels = target2['labels']
    counter = 0
    for box in (target2['boxes']):
        probability = scores[counter]
        if probability > 0.5:
            x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
            rect = patches.Rectangle((x, y),
                                    width, height,
                                    linewidth = 2,
                                    edgecolor = 'r',
                                    facecolor = 'none')

            # Draw the bounding box on top of the image
            a[1].add_patch(rect)
            label_id = predicted_labels[counter]
            a[1].text(x,(y-25), str(list_of_classes[label_id]), verticalalignment='top', color='white',fontsize=10,weight='bold')
            a[1].text(x,(y-10), '%.2f' % probability.item(), verticalalignment='top', color='white',fontsize=10,weight='bold')
        counter += 1

    plt.show()

# plotting the image with bboxes. Feel free to change the index
#img, target = dataset[0]
#plot_img_bbox(img, target)

def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# Send train=True fro training transforms and False for val/test transforms
def get_transform(train):

    if train:
        return A.Compose([
                            A.HorizontalFlip(0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# defining the files directory and testing directory
files_dir = './PCB_DATASET/images_all/'
test_dir ='./PCB_DATASET/images_all/'

# use our dataset and defined transformations
dataset = ImagesDataset(files_dir, 480, 480, transforms= get_transform(train=True))
dataset_test = ImagesDataset(files_dir, 480, 480, transforms= get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

# train test split
test_split = 0.2
tsize = int(len(dataset)*test_split)
dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=5, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=5, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

device='cpu'
num_classes = 7

# get the model using our helper function
model = get_object_detection_model(num_classes)

#load saved weights
checkpoint = torch.load('./checkpoint/ckpt.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# move model to the right device
model.to(device)

def apply_nms(orig_prediction, iou_thresh=0.5):

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')

for i in range(3,6):
    # pick one image from the test set
    img, target = dataset_test[i]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])[0]

    nms_prediction = apply_nms(prediction, iou_thresh=0.5)
    scores = prediction['scores']

    plot_img_bbox(torch_to_pil(img), target,torch_to_pil(img), prediction, scores)
