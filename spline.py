import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.misc import imread, imshow
from scipy import ndimage
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import cv2


def get_g(src, dst):
    result = np.subtract(dst,src)

    return(result)


def spline_interpolation(P,G,Q):
    A = squareform(pdist(P, 'euclidean'))
    B = np.append(np.ones((1, len(P))), np.transpose(P), axis=0)
    C = np.append(np.append(A, np.transpose(B), axis=1), np.append(B, np.zeros((3,3)), axis=1), axis=0)
    D = np.append(G, np.zeros(3))
    wv_ = np.linalg.solve(C,np.transpose(D))
    wv = wv_.reshape((1,len(wv_)))
    dist = cdist(P, Q, 'euclidean')
    X = np.append(dist, np.append(np.transpose(Q), np.ones((1, len(Q))), axis=0) , axis=0)
    result = np.matmul(wv, -X)
    return(result)


def warping(img, inputMark, outputMark):
    height = img.shape[0]
    width = img.shape[1]
    inMark = np.append(inputMark, [[0,0], [0,width-1], [height-1,0], [height-1, width-1]], axis=0)
    outMark = np.append(outputMark, [[0,0], [0,width-1], [height-1,0], [height-1, width-1]], axis=0)
    g = get_g(inMark, outMark)
    P = inMark

    Q_ = list((x, y) for x in range(0, height, 1) for y in range(0, width, 1))
    for i in range(inMark.shape[0] - 1):
        Q_.remove((inMark[i,0], inMark[i,1]))
    Q = np.append(inputMark, np.array(Q_),axis=0)
    Gx = g[:,0]
    Gy = g[:,1]
    X = spline_interpolation(P, Gx, Q)
    Y = spline_interpolation(P, Gy, Q)

    warpingImg = np.empty(img.shape, dtype=np.uint8)
    print(warpingImg.shape)
    resultImg = np.empty(img.shape, dtype=np.uint8)
    for i in range((width-1) * (height-1)):
        x, y = np.clip(int(X[0][i]), 0, width-1), np.clip(int(Y[0][i]),0,height-1)
        warpingImg[y,x] = img[int(Q[i][0]), int(Q[i][1])]

    # bilinear_interp(warpingImg, resultImg, scale)

    return(warpingImg)


def bilinear_interp(img1, img2, rate):
    for y in range(img1.shape[0]):
        for x in range(img1.shape[1]):
            px = int(x/rate)
            py = int(y/rate)
            fx1 = x/rate - px
            fx2 = 1 - fx1
            fy1 = y/rate - py
            fy2 = 1 - fy1

            w1 = fx2 * fy2
            w2 = fx1 * fy2
            w3 = fx2 * fy1
            w4 = fx1 * fy1

            #Get pixels in four corners
            for chan in range(img1.shape[2]):
                bl = img1[py, px, chan]
                br = img1[py, px+1, chan]
                tl = img1[py+1, px, chan]
                tr = img1[py+1, px+1, chan]

                #Calculate interpolation
                img2[y, x, chan] = w1 * bl + w2 * br + w3 * tl + w4 * tr
           
"""
if __name__=="__main__":

    im = imread("test.jpg", mode="RGB")
    plt.imshow(im)
    plt.show()
    warpingImg = np.empty(im.shape, dtype=np.uint8)
    bilinear_interp(im, warpingImg, 5)
    plt.imshow(np.uint8(warpingImg))
    plt.show()
"""
