import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import glob
import operator

count = {}
img = {}

imga = cv2.imread('./test/20151102_131019.jpg',0)
imga2 = imga.copy()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True)
args = vars(ap.parse_args())

# All the 6 methods for comparison in a list
meth = 'cv2.TM_SQDIFF_NORMED'
number = 0
# for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
imagePath = './dataset/20151101_224016.jpg'

img[number] = cv2.imread(imagePath,0)
w, h = img[number].shape[::-1]

# img[number] = img2.copy()
method = eval(meth)

# Apply template Matching
res = cv2.matchTemplate(imga,img[number],method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum

top_left = min_loc

bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(imga,top_left, bottom_right, 255, 2)

plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img[number],cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle(meth)

plt.show()