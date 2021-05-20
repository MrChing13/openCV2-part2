import cv2
from matplotlib import pyplot as plt
import numpy as np

# 1. Draw Line
# 2. Draw Rectangle
# 3. Draw Circle
# 4. Draw Ellipse
# 5. Add Text
# 6. Create Border Image
# 7. Create Corner Detection
# 8. Create Face Detection

# # DRAW LINE
#
# image = cv2.imread("resources/kobe.jpeg")
# cv2.imshow("kobe", image)
#
# start_point = (0, 0)
# end_point = (300, 300)
# color = (255, 0, 0)
# thickness = 5
#
# image = cv2.line(image, start_point, end_point, color, thickness)
#
# cv2.imshow("kobe", image)
# cv2.waitKey(0)

# ------------------------------------------------------
# DRAW RECTANGLE
#
# image = cv2.imread("resources/kobe.jpeg")
# cv2.imshow("kobe", image)
#
# start_point = (50, 50)
# end_point = (210, 210)
# color = (0, 255, 0)
# thickness = 2
#
# image = cv2.rectangle(image, start_point, end_point, color, thickness)
#
# cv2.imshow("kobe", image)
# cv2.waitKey(0)

# ------------------------------------------------------
# # DRAW CIRCLE
#
# image = cv2.imread("resources/kobe.jpeg")
# cv2.imshow("kobe", image)
#
# center_coordinates = (120, 50)
# radius = 20
# color = (0, 0, 255)
# thickness = 2
#
# image = cv2.circle(image, center_coordinates, radius, color, thickness)
#
# cv2.imshow("kobe", image)
# cv2.waitKey(0)

# ------------------------------------------------------
# DRAW ELLIPSE
#
# image = cv2.imread("resources/kobe.jpeg")
# cv2.imshow("kobe", image)
#
# center_coordinates = (120, 100)
# axesLength = (100, 50)
# angle = 0
# startAngle = 0
# endAngle = 360
# color = (0, 0, 255)
# thickness = 5
#
# image = cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)
#
# cv2.imshow("kobe", image)
# cv2.waitKey(0)

# ------------------------------------------------------
# ADD TEXT
# image = cv2.imread("resources/kobe.jpeg")
# cv2.imshow("kobe", image)
#
# text = "KOBE BRYANT"
# coordinates = (30, 50)
# t_font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
# t_scale = 1
# t_color = (255, 0, 100)
# t_thickness = 1
#
# image = cv2.putText(image, text, coordinates, t_font, t_scale, t_color, t_thickness)
#
# cv2.imshow("kobe", image)
# cv2.waitKey(0)

# --------------------------------------------------------
# CREATE BORDER IMAGE
#
# image = cv2.imread("resources/kobe.jpeg")
# cv2.imshow("Kobe", image)
#
# image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
#
# cv2.imshow("Kobe", image)
# cv2.waitKey(0)

# ---------------------------------------
# # CORNER DETECTION

# img = cv2.imread("resources/corner1.png")
# cv2.imshow("shape", img)
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("shape", gray)
#
# corners = cv2.goodFeaturesToTrack(gray, 27, 0.01, 10)
# corners = np.int0(corners)
#
# for i in corners:
#     x, y = i.ravel()
#     cv2.circle(img, (x, y), 3, 255, -1)
#
# plt.imshow(img), plt.show()
# cv2.waitKey(0)

# -------------------------------------------------
# FACE DETECTION

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread("resources/Avengers.jpeg")
cv2.imshow("kobe", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("kobe gray", gray)

faces = faceCascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


cv2.imshow("kobe", img)
cv2.waitKey(0)

