from mtcnn import MTCNN
import cv2

detector = MTCNN()

resl=(1280,720)

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('cropped.mp4')
out = cv2.VideoWriter('out.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25,resl)


while cam.isOpened():
	ret, img = cam.read()
	try:
		img = cv2.resize(img, resl)
		rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	except:
		break
	faces = detector.detect_faces(rgb)
	for idx,bx in enumerate(faces):
		x,y,w,h = bx['box']
		cv2.rectangle(img,(x,y),(w+x,h+y),(0,255,0),2)
		tp = y-15 if y-15 > 15 else y+15
		cv2.putText(img,str(idx+1),(x,tp),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2,cv2.LINE_AA)

	cv2.putText(img,"Count: "+str(len(faces)),(10,60),cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,0,255),4,cv2.LINE_AA)
	cv2.imshow('webcam', img)
	out.write(img)
	if cv2.waitKey(1) & 0xff == 27:
		break

cv2.destroyAllWindows()
cam.release()
out.release()