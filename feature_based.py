import numpy as np
from matplotlib import pyplot as plt
import argparse
import glob
import cv2
import operator
 
count = {}
kp = {}
des = {}
img = {}
matches = {}


img1 = cv2.imread('./test/20151102_200159.jpg',0)   
print "dtype : ", img1.dtype

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True)
args = vars(ap.parse_args())

sift = cv2.xfeatures2d.SIFT_create()


number = 0
kp1,des1 = sift.detectAndCompute(img1,None)

for imagePath in glob.glob(args["dataset"] + "/*.jpg"):

	img[number] = cv2.imread(imagePath,0)
	kp[number],des[number] = sift.detectAndCompute(img[number],None)
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches[number] = flann.knnMatch(des1,des[number],k = 2)
	matchesMask = [[0,0] for i in xrange(len(matches[number]))]
	count[number] = 0
	for i,(m,n) in enumerate(matches[number]):
	    if m.distance < 0.7*n.distance:
	        matchesMask[i]=[1,0]
	        count[number] = count[number] + 1

	print number , count[number]
	draw_params = dict(matchColor = (0,255,0),
	                   singlePointColor = (255,0,0),
	                   matchesMask = matchesMask,
	                   flags = 0)
	number = number + 1

ind = max(count.iteritems(), key=operator.itemgetter(1))[0]
print ind
img3 = cv2.drawMatchesKnn(img1,kp1,img[ind],kp[ind],matches[ind],None,**draw_params)
plt.imshow(img3,),plt.show()
