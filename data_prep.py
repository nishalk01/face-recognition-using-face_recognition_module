
import face_recognition
import cv2
import numpy as np
import os
from PIL import Image
path="dat_data/"
for i,img in enumerate(os.listdir(path)):
    path_to_img=os.path.join(path,img)
    frame=cv2.imread(path_to_img)
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    for j,location in enumerate(face_locations):
      top, right, bottom, left=location
      face_image=rgb_frame[top:bottom, left:right]
      pil_image = Image.fromarray(face_image)
      name="image{}{}.jpg".format(i,j)
      pil_image.save(name)
    






