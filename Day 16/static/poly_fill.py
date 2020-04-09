from __future__ import print_function
import cv2
from pylab import *
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
from skimage.morphology import disk, opening


im = cv2.imread('static/images/mvit.png')
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)

#Stores the pixel locations in variable rows and columns
rows, cols = img.shape
#Whitens the image for preprocessing
white_img = cv2.bitwise_not(np.zeros(im.shape, np.uint8))
#Fill the pixel locations having no values as zero of 8 bits
white_polygon = cv2.bitwise_not(np.zeros(im.shape, np.uint8))
white_gray = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
#Come to middle of image
v = np.median(img)
sigma = 0.33
lower_thresh = int(max(0, (1.0 - sigma) * v))
upper_thresh = int(min(255, (1.0 + sigma) * v))
#Perform Edge Detection
edges = cv2.Canny(img, lower_thresh, upper_thresh)
#Plot the Hough Lines which is y = -cot(theta)*x + r*sin(theta)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)
lines = np.reshape(lines, (lines.shape[0], lines.shape[2]))
#Make clusters of the lines meeting each other
kmeans = KMeans(n_clusters=20).fit(lines)

for line in kmeans.cluster_centers_:
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(white_gray, (x1, y1), (x2, y2), 0, 2)
#Draw the contour lines to join the final set of clusters #detected as obstacles
contours = cv2.findContours(white_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

for cnt in contours:
    cv2.drawContours(white_polygon, cnt, 0, 0, -1)
    man = []
    intense = []
    for col in range(cols):
        for row in range(rows):
            if cv2.pointPolygonTest(cnt, (col, row), False) == 1:
                man.append((row, col))
    for k in man:
        intense.append(im[k])
    intensity = mean(intense)
    if intensity > 170:
       cv2.drawContours(white_polygon, [cnt], 0, 0, -1)


white_gray1 = cv2.cvtColor(white_polygon, cv2.COLOR_BGR2GRAY)
opened = opening(white_gray1, selem=disk(4))
opened = Image.fromarray(opened)
opened.save('result.png')
plt.imshow(opened, cmap='gray')
plt.show()
