import cv2
import os
import pickle
import face_recognition
from imutils import paths

imagePaths = list(paths.list_images("dataset"))

knownEncodings=[]
knownNames=[]

for idx,imagePath in enumerate(imagePaths):
	print("\r",idx+1,"/",len(imagePaths),end="")
	name=imagePath.split(os.path.sep)[-2]

	img = cv2.imread(imagePath)
	rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	boxes = face_recognition.face_locations(rgb,model='hog')
	encodings = face_recognition.face_encodings(rgb,boxes)

	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

print()
data = {"encodings":knownEncodings,"names":knownNames}

with open("encodings.pkl","wb") as f:
	f.write(pickle.dumps(data))