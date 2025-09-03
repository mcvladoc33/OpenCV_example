import cv2

# Load Haar cascade for cat face detection
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")

# Load image
img = cv2.imread("cat2.jpg")
# img = cv2.imread("cat3.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect cat faces with tuned parameters
cats = cat_cascade.detectMultiScale(
    gray,
    scaleFactor=1.03,   # smaller step between scales → more sensitive
    minNeighbors=3,     # lower value → more detections, but also more false positives
    minSize=(40, 40)    # minimum possible object size to detect
)

print("Number of cats detected:", len(cats))

# Draw rectangles around detected cats
for (x, y, w, h) in cats:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Show result
cv2.imshow("Cats detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
