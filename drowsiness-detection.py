import os
import glob as glob
import matplotlib.pyplot as plt
import cv2
import requests
import random
import numpy as np
from PIL import Image

# a script intended to implement live object/drowsiness detection
# originally written in google collab

SEED = 42
np.random.seed(SEED)

!ls

TRAIN = True
# Number of epochs to train for.
EPOCHS = 25

if not os.path.exists("train"):
  !curl -L
  
  class_names = ["Ambulance", "Bus", "Car", "Motocycle", "Truck"]
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# Convert bounding boxes in yolo format into xmin, ymin , xmax, ymax
def yolo2bbox(bboxes):
  xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
  xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1] +bboxes[3]/2
  return xmin, ymin, xmax, ymax 
  
  
  
  def plot_box(image, bboxes, labels):
  # Need the image height and width to denormalize
  # the bounding box coordinates
  h, w, _ = image.shape
  for box_num, box in enumerate(bboxes):
    x1, y1, x2, y2 = yolo2bbox(box)
    # denormalizing the coordinates
    xmin = int(x1*w)
    ymin = int(y1*h)
    xmax = int(x2*w)
    ymax = int(y2*h)
    width = xmax - xmin
    height = ymax - ymin 

    class_name = class_names[int(labels[box_num])]

    cv2.rectangle(
        image,
        (xmin, ymin), (xmax, ymax),
        color=colors[class_names.index(class_name)],
        thickness=2
    )

    font_scale = min(1,max(3,int(w/500)))
    font_thickness = min(2, max(10,int(w/50)))

    p1, p2 = (int(xmin), int(ymin), int(xmax), int(ymax))
    # Text width and height
    tw, th = cv2.getTextSize(
        class_name,
        0, fontScale=font_scake, thickness=font_thickness
    )[0]
    p2 = p1[0] + tw, p1[1] + -th -10
    
  # new imports
  import uuid   # unique identification
  import os
  import time
  
  
  #IMAGES_PATH = os.path.join("data", "images") # /data/images
IMAGES_PATH = r"C:\Users\Have Fun\Documents\coding\python\machine_learning\data\images"
labels = ["awake", "drowsy"]
number_imgs = 20


cap = cv2.VideoCapture(0)
# loop through labels
for label in labels:
  print("Collecting images for {}".format(labels))
  time.sleep(5)

  # loop through images
  for img_num in range(number_imgs):
    print("Collecting images for {}, image_num {}".format(labels, img_num))

    # webcam feed
    ret, frame = cap.read()

    # naming the image file
    imgname = os.path.join(IMAGES_PATH, label+","+str(uuid.uuid1()) +"jpg")

    # write out img to file
    #cv2.imwrite(imgname, frame)
    #cv2.imshow("Image Collection", frame)
    # two second delay between captures
    time.sleep(2)

  if cv2.waitKey(10) & 0xFF == ord("q"):
    break
cap.release()
cv2.destroyAllWindows()


for label in labels:
  print("Collecting images for {}".format(labels))
  for img_num in range(number_imgs):
    print("Collecting images for {}, image_num {}".format(labels, img_num))
    imgname = os.path.join(IMAGES_PATH, label+","+str(uuid.uuid1()) +".jpg")
    print(imgname)
