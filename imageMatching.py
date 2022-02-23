'''
Cpoyright(C) 2020 Peng Shuying
This code is running on client(Ubuntu on TX2). It is used to image registration.
After extracting a set of key feature points by the improved SIFT algorithm, 
three groups of optimal pairs are found according to the ratio of the nearest 
neighbor to the next nearest neighbor, and a homography matrix of the affine 
transformation is generated. Then, one of the images is transformed and stretched, 
so that the two groups of images can realize the registration of recorded image 
positions between different ground objects at different epochs. 
'''
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature
import math
import cv2
from collections import Counter
from PIL import Image
from imageio import imread,imwrite
from skimage.transform import rescale
from skimage.color import rgb2gray
import time

def presessing(image):
    X = [0.0] * 256
    count = 0.0
    height, width = image.shape[:2]
    if(len(image.shape)>2):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    size = (width, height)
    img = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC).astype(np.int16)
    for i in range(height):
        for j in range(width):
            if img[i,j] >= 0 and img[i,j] <= 255:
                X[img[i,j]] += 1.0
                count += 1.0
    if(count != size[0] * size[1]):
        print('Some pixels\' value are not in [0,255]')
    cum = 0.0
    j = 1
    Y = np.arange(256)
    for i in range(256):
        tmp = count
        cum = cum + X[i]
        while(j <= 256 and tmp > abs(count * j / 256.0 - cum)):
           tmp = abs(count * j / 256.0 - cum)
           j = j + 1
        j = j - 1
        Y[i] = j - 1
    for i in range(height):
        for j in range(width):
            a1 = image[i,j]
            image[i,j] = Y[a1]
        '''ix, iy = np.where(image == i)
        for (p,q) in zip(ix, iy):
           image[p,q] = j'''
    return image

def orb_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures = 1500,
             scaleFactor = 2,
             nlevels = 8,
             edgeThreshold = 31,
             firstLevel = 0,
             WTA_K = 4,
             patchSize = 31,
             fastThreshold = 20)
    kp,des = orb.detectAndCompute(gray_image,None)
    #print(len(kp))
    return kp,des

def sift_kp(img):
    if (len(img.shape)>2):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    image = presessing(img)
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des = sift.detectAndCompute(image,None)
    #kp_image = cv2.drawKeypoints(image,kp,None)
    #print(len(kp))
    return kp,des

def surf_kp(image):
    # more efficient
    if (len(image.shape)>2):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image = cv2.GaussianBlur(image,(9,9),0)
    #image = cv2.medianBlur(image,9)
    img = presessing(image)
    #kp, des = cv2.AKAZE_create().detectAndCompute(img, None)
    
    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(2000)
    surf.setExtended(True)
    surf.setUpright(True)
    kp, des = surf.detectAndCompute(img, None)
    #print(kp[1].pt)
    return kp,des

def get_good_match(des1,des2,kp1,kp2):
    _,_,V = np.linalg.svd(des2)
    size_projected = 32
    projector = V[:,:size_projected]
    des2 = des2.dot(projector)  # not for ORB
    des1 = des1.dot(projector)
    #print(des1_new.shape)
    #bf = cv2.BFMatcher()
    #matches = bf.knnMatch(des1, des2, k=2)
    
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    #index_params=dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)  # for ORB
    search_params=dict(checks=20)   # iterating times
    flann=cv2.FlannBasedMatcher(index_params,search_params)
    matches=flann.knnMatch(des1,des2,k=2)  # not for ORB
    
    good = []
    for m, n in matches:
        if m.distance <= 0.5 * n.distance:
            px,py = kp2[m.trainIdx].pt
            #m.distance = m.distance / n.distance
            m.distance = -abs(px-400)-abs(py-400)
            good.append(m)
    return good

def get_best3_match(kp1, goodMatch):
    p = goodMatch[0]   #min1
    px,py = kp1[p.queryIdx].pt
    dis = -10.0
    for match in goodMatch[1:]:
        qx, qy = kp1[match.queryIdx].pt
        a = ((py - qy)**2 + (px - qx)**2)**0.5
        if(dis < a):
            q = match
            dis = a
    a = dis
    dis = -10.0
    qx, qy = kp1[q.queryIdx].pt
    for match in goodMatch[1:]:
        if(match == q):
            continue
        xx, yy = kp1[match.queryIdx].pt
        b = ((py - yy)**2 + (px - xx)**2)**0.5
        c = ((qy - yy)**2 + (qx - xx)**2)**0.5
        tmp = min((b*b+c*c-a*a)/(b*c),\
                  (a*a+c*c-b*b)/(a*c),\
                    (b*b+a*a-c*c)/(b*a))
        if tmp > dis:
            dis = tmp
            r = match
    bestMatch = [p,q,r]
    return bestMatch

def get_best4_match(kp1, goodMatch):
    p = goodMatch[0]   #min1
    px,py = kp1[p.queryIdx].pt
    dis = -10.0
    for match in goodMatch[1:]:
        xx, yy = kp1[match.queryIdx].pt
        a = ((py - yy)**2 + (px - xx)**2)**0.5
        if(dis < a):
            q = match
            dis = a
    a = dis
    dis = -10.0
    qx, qy = kp1[q.queryIdx].pt
    for match in goodMatch[1:]:
        if(match == q):
            continue
        xx, yy = kp1[match.queryIdx].pt
        b = ((py - yy)**2 + (px - xx)**2)**0.5
        c = ((qy - yy)**2 + (qx - xx)**2)**0.5
        tmp = min((b*b+c*c-a*a)/(b*c),\
                  (a*a+c*c-b*b)/(a*c),\
                    (b*b+a*a-c*c)/(b*a))
        if tmp > dis:
            dis = tmp
            r = match
    dis = -10.0
    rx, ry = kp1[r.queryIdx].pt
    for match in goodMatch[1:]:
        if(match in [q,r]):
            continue
        xx, yy = kp1[match.queryIdx].pt
        a = min(((py - yy)**2 + (px - xx)**2),\
            ((qy - yy)**2 + (qx - xx)**2),\
            ((ry - yy)**2 + (rx - xx)**2))
        if(dis < a):
            s = match
            dis = a
    bestMatch = [p,q,r,s]
    return bestMatch

def siftImageAlignment(image1,image2):
    height, width = image1.shape[:2]
    log = open('log.txt','a')
    start_total = time.time()
    old_size = (width, height)
    if(image2.shape != image1.shape):
        image2 = cv2.resize(image2, old_size, interpolation=cv2.INTER_CUBIC)
    size = (800,800)
    img1 = cv2.resize(image1, size, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(image2, size, interpolation=cv2.INTER_CUBIC)
    
    kp1,des1 = surf_kp(img1)
    kp2,des2 = surf_kp(img2)
    goodMatch = get_good_match(des1,des2,kp1,kp2)

    start = time.time()
    imgOut=[]
    H = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
    status=[]
    # print(len(goodMatch))
    if len(goodMatch) >= 3:
        goodMatch = sorted(goodMatch,key=lambda x:x.distance)
        bestMatch = get_best3_match(kp1, goodMatch)
        #bestMatch = goodMatch[:3]  #min3
        
        ptsA = np.float32([kp1[m.queryIdx].pt for m in bestMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in bestMatch]).reshape(-1, 1, 2)
        #ransacReprojThreshold = 5.0
        #H, status = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
        H = cv2.getAffineTransform(ptsA,ptsB)
        if __name__ == "__main__":
            img3 = cv2.drawMatches(img1,kp1,img2,kp2,bestMatch,None, flags=2)
            plt.imshow(img3),plt.show()
            print(H)
    imgOut = cv2.warpAffine(img2, H, size,flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
    #imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
    imgOut = cv2.resize(imgOut, old_size, interpolation=cv2.INTER_CUBIC)
    
    end_total = time.time()
    print("time of image_matching cost",end_total-start_total,'s')
    log.write("time of image_matching cost"+str(end_total-start_total)+' s\n')
    log.close()
    return imgOut,H,status

def analyzing(img1, img2):
    x = [0] * 256
    count = 0
    avg = 0.0
    entropy = 0.0
    rmse = 0.0
    #print(img1.shape)
    height, width = img1.shape[:2]
    if(len(img1.shape)>2):
        size = (width, height)
        image1 = cv2.resize(img1, size, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(img2, size, interpolation=cv2.INTER_CUBIC)
        img1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

    img1 = np.array(img1).astype(np.int16)
    img2 = np.array(img2).astype(np.int16)
    #diff = abs(img1-img2)

    for i in range(height):
        for j in range(width):
            if img1[i,j] >= 1 and img2[i,j] >= 1 and img1[i,j] <= 254 and img2[i,j] <= 254:
                diff = abs(img1[i,j] - img2[i,j])
                x[diff] += 1
                avg = avg + diff
                count += 1
    avg = avg / count
    #print("avg=%d   count=%d" %(avg, count))
    
    for i in range(256):
        if x[i]>0:
            entropy = entropy - (x[i]/count) * math.log(x[i]/count)
            rmse = rmse + x[i]/count*(i-avg)**2
    print("entropy is %.4f, rmse is %.4f" %(entropy, math.sqrt(rmse)))
    
def image_matching(path1, path2):
    type1 = path1.split('.')[-1]
    type2 = path2.split('.')[-1]
    img1 = imread(path1,format=type1)
    img2 = imread(path2,format=type2)
    result,_,_ = siftImageAlignment(img1,img2)
    #allImg = np.concatenate((img1,img2,result),axis=1)
    #cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
    #cv2.imshow('Result',allImg)
    #cv2.waitKey(0)
    savepath = path2.split(type2)[0]+path1.split('/')[-1]
    imwrite(savepath, result)
    #analyzing(img1,result) 
    return savepath

if __name__ == "__main__":
    a = './ElephantButte_08201991.jpg'
    b = './ElephantButte_08272011.jpg'
    a1 = './001a.jpg'
    a2 = './001b.jpg'
    b1 = './Andasol_09051987.jpg'
    b2 = './Andasol_09122013.jpg' 
    tst1 = './1599621322280.jpg'
    tst2 = './1599621322244.jpg'
    image_matching(a1,a2)

