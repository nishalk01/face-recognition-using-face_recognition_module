import os
import cv2
import face_recognition
import numpy as np
import pickle
path_to_unknown="abhi/"
path="local_data/prep_processed/"
faces=[]
face_name=[]
files=os.listdir(path)

for file_ in files:
    folder=os.path.join(path,file_)
    folders=os.listdir(folder)
    for img_name in folders:
        path_to_img=os.path.join(folder,img_name)
        print(path_to_img)
        image=face_recognition.load_image_file(path_to_img)
        encoding=face_recognition.face_encodings(image)[0]
        print(len(encoding))
        get_file(encoding,path_to_img)
        name=os.path.dirname(path_to_img)
        base_folder=os.path.basename(name)
        faces.append(encoding)
        face_name.append(base_folder)



#save as pickle file the face_encoding and names
pickle_out=open("x.pickle","wb")
pickle.dump(faces,pickle_out)
pickle_out.close()

pickle_out=open("y.pickle","wb")
pickle.dump(face_name,pickle_out)    
pickle_out.close()

#once the pickle is saved uncomment the above code except modules import and path
pickle_in = open("x.pickle","rb")
faces = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
face_name = pickle.load(pickle_in)

for i,img in enumerate(os.listdir(path_to_unknown)):
    path_to_img_unknown=os.path.join(path_to_unknown,img)
    image=face_recognition.load_image_file(path_to_img_unknown)
    face_locations = face_recognition.face_locations(image)
    face_encodings=face_recognition.face_encodings(image,face_locations)
    for face_encoding,location in zip(face_encodings,face_locations):
      matches = face_recognition.compare_faces(faces, face_encoding,0.4)
      face_distances = face_recognition.face_distance(faces, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = face_name[best_match_index]
      if True in matches:
        top, right, bottom, left=location
        cv2.rectangle(image,(left,top),(right,bottom),(0,255,0),2)
        cv2.rectangle(image, (left, bottom - 10), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(image,name,(left+4,bottom-4),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,0),1)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imshow('s',image)
    name="name{}.jpg".format(i)
    cv2.imwrite(name,image)
    cv2.waitKey(0)
      
    