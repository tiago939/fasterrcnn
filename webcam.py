import os, sys
import random
import numpy as np
import pandas as pd
import cv2
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
import utils
import time


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

device='cpu'
num_classes = 7

def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# get the model using our helper function
model = get_object_detection_model(num_classes)

#load saved weights
#checkpoint = torch.load('./checkpoint/ckpt.pth')
#model.load_state_dict(checkpoint['model_state_dict'])

# move model to the right device
model.to(device)
model = model.float()

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

model.eval()
#STREAM WEBCAM
video = cv2.VideoCapture(0);
while True:
    t0 = time.time()
    check, frame = video.read();
    print('t0 ', time.time()-t0)

    t1 = time.time()
    img = frame/255
    img = torch.tensor(img.transpose((2,0,1))).float()
    print('t1 ', time.time() - t1)

    t2 = time.time()
    with torch.no_grad():
        prediction = model([img.to(device)])[0]
    print('t2 ', time.time() - t2)

    t3 = time.time()

    frame = 255*img.numpy().transpose((1,2,0))
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    for i in range(len(prediction['boxes'])):
        score = prediction['scores'][i].item()
        if score > 0.0:
            x = int(prediction['boxes'][i][0].item())
            y = int(prediction['boxes'][i][1].item())
            w = int(prediction['boxes'][i][2].item())
            h = int(prediction['boxes'][i][2].item())
            frame = cv2.rectangle(frame, (x,y), (w,h), (0,255,0), 3)

    print('t3 ', time.time() - t3)
    cv2.imshow('Face Detector', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break;

video.release()
cv2.destroyAllWindows()

