#================================================================
#
#   File name   : object_tracker.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : code to track detected object from video or webcam
#
#================================================================
# NOTE: To run this, please run
# pip install googledrivedownloader
# pip install pillow
# in your terminal

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *
import time
import random
import requests, re
import torch, torchvision
from torch import nn, optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from google_drive_downloader import GoogleDriveDownloader as gdd
from PIL import Image, ImageFont, ImageDraw 

video_path   = ""

parser = argparse.ArgumentParser()
parser.add_argument('--images', action='store_true', help = "Use a folder of images instead of video feed")
args = parser.parse_args()

# Preparing the Training Dataset
# UNCOMMENT THE TWO LINES BELOW IF RUNNING THIS SCRIPT FOR THE FIRST TIME
# Dataset (Small)
#gdd.download_file_from_google_drive(file_id='1BEi1Cqi8yE9JJ9s3HPGJFBAi038MHjIc', dest_path='./data/classifier/standing/standing.zip', unzip=True)
#gdd.download_file_from_google_drive(file_id='1z0f0uemZt4gdcp9mimnjI8TkYaVD-oYG', dest_path='./data/classifier/sitting/sitting.zip', unzip=True)

# Dataset (Medium)
#gdd.download_file_from_google_drive(file_id='16P91uG96lxhXB5kQvj7sxJdNCphKuNgk', dest_path='./data/classifier/standing/standing.zip', unzip=True)
#gdd.download_file_from_google_drive(file_id='1M20vI-3iOxixmJ4BRgM_Xk5GiQ2A6GHg', dest_path='./data/classifier/sitting/sitting.zip', unzip=True)

xform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
resize = transforms.Resize((224,224))
dataset_full = datasets.ImageFolder('data/classifier', transform=xform)

# Preparing for labeling
label_font = ImageFont.truetype('Font/Calibri.ttf', 14)


############# Preparing the Model and Dataset for Training #############
n_all = len(dataset_full)
n_train = int(0.8 * n_all)
n_test = n_all - n_train
rng = torch.Generator().manual_seed(2048)
dataset_train, dataset_test = torch.utils.data.random_split(dataset_full, [n_train, n_test], rng)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = 4, shuffle=True)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 4, shuffle=True)

# Use the line below for NVIDIA GPU
# device = torch.device('cuda:0')
# Use the line below for CPU only
device = torch.device("cpu")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
torch.nn.init.xavier_uniform_(model.fc.weight)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

xform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
dataset_full = datasets.ImageFolder('data/classifier', transform=xform)

########################################################################

def run_test(model):
    nsamples_test = len(dataset_test)
    loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for samples, labels in loader_test:
            samples = samples.to(device)
            labels = labels.to(device)
            outs = model(samples)
            loss += criterion(outs, labels)
            _, preds = torch.max(outs.detach(), 1)
            correct_mask = preds == labels
            correct += correct_mask.sum(0).item()
    return loss / nsamples_test, correct / nsamples_test

def run_train(model, opt, sched):
    nsamples_train = len(dataset_train)
    loss_sofar, correct_sofar = 0, 0
    model.train()
    with torch.enable_grad():
        for samples, labels in loader_train:
            samples = samples.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            outs = model(samples)
            _, preds = torch.max(outs.detach(), 1)
            loss = criterion(outs, labels)
            loss.backward()
            opt.step()
            loss_sofar += loss.item() * samples.size(0)
            correct_sofar += torch.sum(preds == labels.detach())
    sched.step()
    return loss_sofar / nsamples_train, correct_sofar / nsamples_train

def run_all(model, optimizer, scheduler, n_epochs):
    for epoch in range(n_epochs):
        loss_train, acc_train = run_train(model, optimizer, scheduler)
        loss_test, acc_test = run_test(model)
        #print(f"epoch {epoch}: train loss {loss_train:.4f} acc {acc_train:.4f}, test loss {loss_test:.4f} acc {acc_test:.4f}")

def Object_tracking(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only = []):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1) #original batch_size
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times, times_2 = [], []
    
    if args.images == True:
        use_image = True
        print("Reading images")
    else:
        use_image = False
        print("Getting webcam feed")

    if use_image == True:
        image_folder_path = 'IMAGES/'
	# image_folder_path = '/home/matthew/Desktop/PersonTracking/IMAGES/'
    elif video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(-1) # detect from webcam

    if use_image == False:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())
    if (use_image == True):
      for image_path in os.listdir(image_folder_path):
          input_path = os.path.join(image_folder_path,image_path)
          frame = cv2.imread(input_path)
          try:
              original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
          except:
              break
          
          image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
          #image_data = tf.expand_dims(image_data, 0)
          image_data = image_data[np.newaxis, ...].astype(np.float32)

          t1 = time.time()
          if YOLO_FRAMEWORK == "tf":
              pred_bbox = Yolo.predict(image_data)
          elif YOLO_FRAMEWORK == "trt":
              batched_input = tf.constant(image_data)
              result = Yolo(batched_input)
              pred_bbox = []
              for key, value in result.items():
                  value = value.numpy()
                  pred_bbox.append(value)
          
          #t1 = time.time()
          #pred_bbox = Yolo.predict(image_data)
          t2 = time.time()
          
          pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
          pred_bbox = tf.concat(pred_bbox, axis=0)

          bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
          bboxes = nms(bboxes, iou_threshold, method='nms')

          # extract bboxes to boxes (x, y, width, height), scores and names
          boxes, scores, names = [], [], []
          for bbox in bboxes:
              if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                  boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                  scores.append(bbox[4])
                  names.append(NUM_CLASS[int(bbox[5])])

          # Obtain all the detections for the given frame.
          boxes = np.array(boxes) 
          names = np.array(names)
          scores = np.array(scores)
          features = np.array(encoder(original_frame, boxes))
          detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in 
  zip(boxes, scores, names, features)]

          # Pass detections to the deepsort object and obtain the track information.
          tracker.predict()
          tracker.update(detections)

          # Obtain info from the tracks
          tracked_bboxes = []
          for track in tracker.tracks:
              if not track.is_confirmed() or track.time_since_update > 5:
                  continue 
              bbox = track.to_tlbr() # Get the corrected/predicted bounding box
              class_name = track.get_class() #Get the class name of particular object
              tracking_id = track.track_id # Get the ID for the particular track
              index = key_list[val_list.index(class_name)] # Get predicted object index by object name
              tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

          # draw detection on frame
          image = draw_bbox(original_frame.copy(), tracked_bboxes, CLASSES=CLASSES, tracking=True)

          # draw original yolo detection
          #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)

          path = 'output_images/'         
          # path = '/home/matthew/Desktop/PersonTracking/output_images/'
          allowed_class = "person"
          num_objects = len(names)
          # classes = .................................
          # create dictionary to hold count of objects for image name
          counts = dict()
          for i in range(num_objects):
              # get count of class for part of image name
              if names[i] == allowed_class:
                counts[i] = counts.get(i, 0) + 1
		# get box coords
                x, y, w, h = boxes[i]
                # crop detection from image (take an additional 5 pixels around all edges)
                cropped_img = original_frame.copy()[y:y+h,x:x+w,:]                
                # construct image name and join it to path for saving crop properly
                
                # Get pose estimation and draw the label
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                cropped_img = Image.fromarray(cropped_img)
                cropped_img = resize(cropped_img)
                cropped_img_for_model = xform(cropped_img)
                cropped_img_for_model = cropped_img_for_model.unsqueeze(0)
                pose_prediction = model(cropped_img_for_model)
                _, preds = torch.max(pose_prediction, 1)      
                if preds == 0:
                    label = "Sitting"
                else:
                    label = "Standing"
                image_editable = ImageDraw.Draw(cropped_img)
                image_editable.text((15,15), label, (0, 252, 76), font=label_font)

                #img_name = 'person' + '_' + str(counts[i]) + '.png'
                img_name = 'person' + '_' + str(random.sample(range(1000000), 1)) + '.png'
                path = 'output_images/' 
                img_out_path = os.path.join(path, img_name )              
                # save image

                cropped_img.save(img_out_path, 'PNG')
                #cv2.imwrite(img_out_path, cropped_img)
              else:
                continue

          if show:
              cv2.imshow('output', image)
              
              if cv2.waitKey(25) & 0xFF == ord("q"):
                  cv2.destroyAllWindows()
                  break
      # End of Image While Loop
######### Code below uses a live video feed ##############################
    if (use_image == False):
      while True:
          _, frame = vid.read()
          try:
              original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
          except:
              break
          
          image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
          #image_data = tf.expand_dims(image_data, 0)
          image_data = image_data[np.newaxis, ...].astype(np.float32)

          t1 = time.time()
          if YOLO_FRAMEWORK == "tf":
              pred_bbox = Yolo.predict(image_data)
          elif YOLO_FRAMEWORK == "trt":
              batched_input = tf.constant(image_data)
              result = Yolo(batched_input)
              pred_bbox = []
              for key, value in result.items():
                  value = value.numpy()
                  pred_bbox.append(value)
          
          #t1 = time.time()
          #pred_bbox = Yolo.predict(image_data)
          t2 = time.time()
          
          pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
          pred_bbox = tf.concat(pred_bbox, axis=0)

          bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
          bboxes = nms(bboxes, iou_threshold, method='nms')

          # extract bboxes to boxes (x, y, width, height), scores and names
          boxes, scores, names = [], [], []
          for bbox in bboxes:
              if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                  boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                  scores.append(bbox[4])
                  names.append(NUM_CLASS[int(bbox[5])])

          # Obtain all the detections for the given frame.
          boxes = np.array(boxes) 
          names = np.array(names)
          scores = np.array(scores)
          features = np.array(encoder(original_frame, boxes))
          detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in 
  zip(boxes, scores, names, features)]

          # Pass detections to the deepsort object and obtain the track information.
          tracker.predict()
          tracker.update(detections)

          # Obtain info from the tracks
          tracked_bboxes = []
          for track in tracker.tracks:
              if not track.is_confirmed() or track.time_since_update > 5:
                  continue 
              bbox = track.to_tlbr() # Get the corrected/predicted bounding box
              class_name = track.get_class() #Get the class name of particular object
              tracking_id = track.track_id # Get the ID for the particular track
              index = key_list[val_list.index(class_name)] # Get predicted object index by object name
              tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

          # draw detection on frame
          image = draw_bbox(original_frame.copy(), tracked_bboxes, CLASSES=CLASSES, tracking=True)

          t3 = time.time()
          times.append(t2-t1)
          times_2.append(t3-t1)
          
          times = times[-20:]
          times_2 = times_2[-20:]

          ms = sum(times)/len(times)*1000
          fps = 1000 / ms
          fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
          
          image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

          # draw original yolo detection
          #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)

          # path = '/home/matthew/Desktop/PersonTracking/output_images/'
          path = 'output_images/'
          allowed_class = "person"
          num_objects = len(names)
          # classes = .................................
          # create dictionary to hold count of objects for image name
          counts = dict()
          for i in range(num_objects):
              # get count of class for part of image name
              if names[i] == allowed_class:
                counts[i] = counts.get(i, 0) + 1
                # get box coords
                x, y, w, h = boxes[i]
                # crop detection from image (take an additional 5 pixels around all edges)
                cropped_img = original_frame.copy()[y:y+h,x:x+w,:]
                # construct image name and join it to path for saving crop properly

                # Get pose estimation and draw the label
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                cropped_img = Image.fromarray(cropped_img)
                cropped_img = resize(cropped_img)
                cropped_img_for_model = xform(cropped_img)
                cropped_img_for_model = cropped_img_for_model.unsqueeze(0)
                pose_prediction = model(cropped_img_for_model)
                _, preds = torch.max(pose_prediction, 1)      
                if preds == 0:
                    label = "Sitting"
                else:
                    label = "Standing"
                image_editable = ImageDraw.Draw(cropped_img)
                image_editable.text((15,15), label, (0, 252, 76), font=label_font)

                #img_name = 'person' + '_' + str(counts[i]) + '.png'
                img_name = 'person' + '_' + str(random.sample(range(1000000), 1)) + '.png'
                img_out_path = os.path.join(path, img_name )              
                # save image
                cropped_img.save(img_out_path, 'PNG')
                #cv2.imwrite(img_path, cropped_img)
              else:
                continue

          print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
          if output_path != '': out.write(image)
          if show:
              cv2.imshow('output', image)
              
              if cv2.waitKey(25) & 0xFF == ord("q"):
                  cv2.destroyAllWindows()
                  break
########### End of code for live video feed ########################


    cv2.destroyAllWindows()


yolo = Load_Yolo_model()
#run_all(model, optimizer, scheduler, 10)
model_path = 'model.pth'
#torch.save(model.state_dict(), model_path)
model.load_state_dict(torch.load(model_path))
model.eval()
Object_tracking(yolo, video_path, "detection.mp4", input_size=YOLO_INPUT_SIZE, show=True, iou_threshold=0.1, rectangle_colors=(255,0,0), Track_only = ["person"])
