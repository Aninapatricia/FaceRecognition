# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:00:00 2019

@authors: Tania, Nadia and Anina
Work based on #https://medium.com/data-science-lab-amsterdam/face-recognition-with-python-in-an-hour-or-two-d271324cbeb3
"""

import face_recognition
import cv2
import numpy as np
import glob
import os
import logging
IMAGES_PATH = './Database';
MAX_DISTANCE = 0.6 # increase to make recognition less strict, decrease to make more strict
MAX_DISTANCE = 0.7 # increase to make recognition less strict, decrease to make more strict
WINDOW_NAME = "Face ID"

# ---   CAMERA SETTINGS   ---
VIDEO_CAPTURE = None
DEVICE_INDEX  = 0
EXPOSURE_TIME = -7
FLIP_IMAGE = True

# --- define dataset ---
names = {}


def change_cam(idxOffset):  
    global VIDEO_CAPTURE
    global DEVICE_INDEX
    idx = DEVICE_INDEX + idxOffset
    cam = None
    cam = cv2.VideoCapture(idx) 
    if not cam.isOpened():
        return -1
    ret, frame = cam.read()  
    if ret==False:
        cam.release()
        return -1
    VIDEO_CAPTURE = cam
    DEVICE_INDEX = idx
    return 0

def change_exposure(expOffset):  
    global VIDEO_CAPTURE
    global EXPOSURE_TIME
    expTime = EXPOSURE_TIME + expOffset
    if(expOffset == 0):
        try:
            # VIDEO_CAPTURE.set('ExposureMode','auto')
            VIDEO_CAPTURE.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
        except:
            return
    else:
        try:
            #VIDEO_CAPTURE.set('Exposure',expTime)
            VIDEO_CAPTURE.set(cv2.CAP_PROP_AUTO_EXPOSURE,0)
            VIDEO_CAPTURE.set(cv2.CAP_PROP_EXPOSURE,expTime)
            EXPOSURE_TIME = expTime
            setExpTime = VIDEO_CAPTURE.get(cv2.CAP_PROP_EXPOSURE)
            if(expTime < setExpTime):
                EXPOSURE_TIME = setExpTime
            if(expTime > setExpTime):
                EXPOSURE_TIME = setExpTime
        except:
            return
            

def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)

    # run the embedding model to get face embeddings for the supplied locations
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings


class MyDatabase():
    """
    Load reference images and create a database of their face encodings
    """
    def __init__(self):
        features = {}
        names = {}
    
        for foldername in glob.glob(os.path.join(IMAGES_PATH, '*')):
            fn = os.path.split(foldername)[1]
            person = 'unknown'
            # for filename in glob.glob(os.path.join(foldername, '*.name')): # *.jpg)):
            #    person = os.path.splitext(os.path.basename(filename))[0]
            person = os.path.splitext(os.path.basename(foldername))[0]
            for filename in glob.glob(os.path.join(foldername, '*.*')): # *.jpg)):
                # load image
                image = cv2.imread(filename)
                if( image is None ):
                    continue
                image_rgb = face_recognition.load_image_file(filename)
        
                # use the name in the filename as the identity key
                identity = fn + '_' + os.path.splitext(os.path.basename(filename))[0]
        
                # get the face encoding and link it to the identity
                locations, encodings = get_face_embeddings_from_image(image_rgb)
                if len(encodings)>0:
                    features[identity] = encodings[0]
                    if (person == 'unknown'):
                        names[identity] = fn
                    else:
                        print(person)
                        names[identity] = person
    
        self.features = features
        self.names = names


def paint_detected_face_on_image(frame, location, name=None):
    """
    Paint a rectangle around the face and write the name
    """
    # unpack the coordinates from the location tuple
    top, right, bottom, left = location
    print(location)

    if name is None:
        name = 'Unknown'
        color = (0, 0, 255)  # red for unrecognized face
    else:
        color = (0, 128, 0)  # dark green for recognized face

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)


def run_face_recognition(features, names):
    """
    Start the face recognition via the webcam
    """
    global VIDEO_CAPTURE, FLIP_IMAGE
    bFullScreen = 0   # 1: start in FullScreen mode
    
    # Open a handler for the camera
    # Start capturing the WebCam  
    VIDEO_CAPTURE = cv2.VideoCapture(DEVICE_INDEX)

    VIDEO_CAPTURE.set(3,240);
    VIDEO_CAPTURE.set(4,240);    

    # the face_recognitino library uses keys and values of your database separately
    known_face_encodings = list(features.values())
    known_face_names = list(names.values())
    
    hWnd = cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, bFullScreen)
    
    while VIDEO_CAPTURE.isOpened():
        
        # Grab a single frame of video (and check if it went ok)
        ok, frame = VIDEO_CAPTURE.read()
        if not ok:
            logging.error("Could not read frame from camera. Stopping video capture.")
            break
                
        # run detection and embedding models
        if(FLIP_IMAGE):
            frame = cv2.flip(frame, +1);
                
        # run detection and embedding models
        face_locations, face_encodings = get_face_embeddings_from_image(frame, convert_to_rgb=True)
        # Loop through each face in this frame of video and see if there's a match
        for location, face_encoding in zip(face_locations, face_encodings):

            # get the distances from this encoding to those of all reference images
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # select the closest match (smallest distance) if it's below the threshold value
            if np.any(distances <= MAX_DISTANCE):
                best_match_idx = np.argmin(distances)
                name = known_face_names[best_match_idx]
            else:
                name = None

            # put recognition info on the image
            paint_detected_face_on_image(frame, location, name)

        # Display the resulting image
        cv2.imshow(WINDOW_NAME, frame)

        ch = cv2.waitKeyEx(1)
        # Test for fullscreen toggle (F11): Qt::Key_F11
        if ch == 0x7A0000: # 0x7A0000 = F11 on Windows
            bFullScreen = 1-bFullScreen
            cv2.setWindowProperty( WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, bFullScreen)
        ch = ch & 0xFF
        if ch == 27:
    		   break
        if ch == ord(','):
            change_exposure(0)
        if ch == ord('.'):
            change_exposure(1)
        if ch == ord('-'):
    	      change_exposure(-1)
        if ch == ord('f'):
    		   FLIP_IMAGE = not FLIP_IMAGE
        if ch == ord('n'):
    		   change_cam(1)
        if ch == ord('p'):
    	      change_cam(-1)
        if ch == ord('q'):  
    	      break  

    # Release handle to the webcam
    VIDEO_CAPTURE.release()
    cv2.destroyAllWindows()
    
def main():
    database = MyDatabase()
    features = database.features
    names = database.names
    run_face_recognition(features, names)
    
    
if __name__ == "__main__":
    main()
    

