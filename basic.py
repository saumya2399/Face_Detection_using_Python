import cv2

# create a cascade classifier object
# face_cascade = cv2.CascadeClassifier("C:\\Users\\Dell\\PycharmProjects\\pythonProject\\man.jpg")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# reading the image as it is
img = cv2.imread("C:\\Users\\Dell\\PycharmProjects\\pythonProject\\man.jpg")

# reading the image as gray scale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# search the coordinates of the image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

print(type(faces))
print(faces)

for x,y,w,h in faces:
    img=cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

resized = cv2.resize(img, (int(img.shape[1]),int(img.shape[0])))

cv2.imshow("Gray",img)
cv2.waitKey(0)
cv2.destroyAllWindows()