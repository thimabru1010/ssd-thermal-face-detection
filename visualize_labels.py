import cv2
import os
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
#from matplotlib import pyplot as plt

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def detect_dnn_frame(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
    conf_threshold = 0.7

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def show_labels(folder_path):

    data = pd.read_csv(os.path.join(folder_path,'sub-'+'train'+'-annotations-bbox.csv')).values.tolist()
    # teste = np.array(annot[['Set', 'Participant', 'File']].astype(str))
    # print(teste)
    # print('='*60)
    # folders.sort(key=int)

    train_folder =os.path.join(folder_path, 'train')
    image_names = os.listdir(train_folder)
    image_names.sort()
    #print(image_names)
    for d in data:
    #for i, image_name in enumerate(image_names):
        print(d)
        image_id = str(d[0])
        image_path = os.path.join(train_folder, image_id)
        print(image_path)
        thermal = cv2.imread(image_path)
        print(thermal.shape)

        #print(image_path.split('/')[1:])
        #(x,y,w,h) = np.array(d[['XMin', 'XMax', 'YMin', 'YMax']])
        #(x,y,w,h) = (d[1], d[2], d[3], d[4])
        (x,y,w,h) = (d[1], d[2], d[3], d[4])
        print((x,y,w,h))
        #thermal = cv2.rectangle(thermal,(x,y),(x+w,y+h),(255,0,0),2)
        thermal = cv2.rectangle(thermal,(x,y),(w,h),(255,0,0),2)
        cv2.imshow('Thermal', thermal)


        if cv2.waitKey(0) > 0:
            continue
            #break

        # cv2.imshow('Original', img)
        # cv2.imshow('Cinza', gray)
        # cv2.waitKey(0)

folder_path = 'data/CelebA/img_celeba_splitted'
#folder_path = 'data/Thermal_organized_splitted'
show_labels(folder_path)
#match_template(folder_path, rgb_folder_path)
