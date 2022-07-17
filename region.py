import cv2
import numpy as np


def Kmeans_cluster(img,K):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #changing 2D shape to 1D
    twoDimage = img.reshape((-1, 3))  # height*width, color channel
    twoDimage = np.float32(twoDimage)

    #for selecting k no of centers itreating 10 times
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    attempts = 10

    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    #falting 2D level in 1D
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image


