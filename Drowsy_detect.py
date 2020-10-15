#!/usr/bin/env python
# coding: utf-8

# In[67]:


from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from collections import OrderedDict
import sounddevice
import matplotlib
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


# In[68]:


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
#ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())


# In[ ]:





# In[ ]:





# In[ ]:


def find_ear(eye):
 A = dist.euclidean(eye[1], eye[5])
 B = dist.euclidean(eye[2], eye[4])
 C = dist.euclidean(eye[0], eye[3]) 
 ear = (A + B) / (2.0 * C)
 return ear


# In[65]:


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


# In[66]:


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# In[53]:


print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
time.sleep(1.0)


# In[ ]:




