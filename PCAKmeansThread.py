'''
Cpoyright(C) 2020 Peng Shuying
This code is running on client(Ubuntu on TX2). It is used to generate the 
difference image. In order to improve the processing efficiency, the 
differential image is sampled down, and then a feature vector of pixel 
blocks in the differential image is extracted via PCA. By projecting the 
neighborhood of each pixel in the differential image onto the feature 
vector, a feature vector of pixels is constructed. In this feature space, the 
k-means algorithm is used to distinguish between the two classes, and 
finally one pixel is obtained to represent the class of change and the other 
to represent the pixels belonging to the invariant class. A morphological 
operation of the binary gray image after clustering is carried out to 
eliminate any isolated noise points which were mistakenly clustered, and 
the final change detection results are obtained.
'''
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from PIL import Image
from imageio import imread,imwrite
from skimage.color import rgb2gray
import time
from threading import Thread,Lock

lock = Lock()
class myThread(Thread):
    def __init__(self,threadID,name='default'):
        Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def getfunc(self,func=None):
        self.func = func
 
    def getargs(self, **kwargs):
        self.kwargs = kwargs       

    def run(self):
        if lock.acquire():
            if self.func == None:
                self.output = self.kwargs
            else:
                self.output = self.func(**self.kwargs)

            lock.release()

    def getresult(self):
        return self.output

def find_vector_set(diff_image, new_size):
    vector_set = np.zeros((new_size[0] * int(new_size[1] / 5), 25))   #5*5
    j = 0
    for i in range(vector_set.shape[0]):
        while j < new_size[0]:
            k = 0
            while k < new_size[1]:
                block = diff_image[j:j+5, k:k+5]
                #print(i,j,k,block.shape)
                vector_set[i, :] = block.ravel()
                k = k + 5
            j = j + 5  

    mean_vec   = np.mean(vector_set, axis = 0)    
    vector_set = vector_set - mean_vec
    #print("vector_set shape",vector_set.shape,mean_vec.shape)
    return vector_set, mean_vec
    
def find_FVS(EVS, diff_image, mean_vec, new):
    start_vs = time.time()
    N_pixel = new[0] * new[1] - 4 * new[0] - 4 * new[1] + 16
    feature_vector_set = np.zeros((N_pixel, 25))
    FVS = np.zeros((N_pixel, 25))
    for k in range(-2, 3, 1):
        for kk in range(-2, 3, 1):
            index_1_k = -2 + k if k != 2 else None
            index_1_kk = -2 + kk if kk != 2 else None
            tem_array = diff_image[2 + k: index_1_k, 2 + kk: index_1_kk]
            index_FV = (k + 2) * 5 + kk + 2
            feature_vector_set[:, index_FV] = tem_array.flatten()
    end_vs = time.time()
    log = open('log.txt','a')
    print('runtime of vector space: ' + str(end_vs - start_vs) + ' seconds')
    log.write('runtime of vector space: ' + str(end_vs - start_vs) + ' seconds\n')

    start_fvs = time.time()
    EVS = np.array(EVS)
    n = 12
    thr = []
    length = int(N_pixel / n)
    for i in range(n):
        up = i * length
        down = (i+1)*length if i<n-1 else N_pixel
        t = myThread(i,"Thread-"+str(i))
        t.getfunc()
        t.getargs(vector = feature_vector_set[up:down,:],\
                  up=up,down=down)
        thr.append(t)
    for t in thr:
        t.start()
        t.join()
        FVS[t.kwargs['up']:t.kwargs['down'],:] = np.dot(t.kwargs['vector'], EVS)
            
    #FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    #print("\nfeature vector space size",FVS.shape)
    end_fvs = time.time()
    print("runtime of feature vector space:t",end_fvs-start_fvs,'s')
    log.write('runtime of vector space: ' + str(end_vs - start_vs) + ' s\n')
    log.close()
    return FVS

def clustering(FVS, components, new):
    start = time.time()
    kmeans = KMeans(components,max_iter=20,precompute_distances=True,\
                    init='k-means++',n_jobs=-1,n_init=1)
    kmeans.fit(FVS)
    end = time.time()
    log = open('log.txt','a')
    print("time of clustering cost:(part 1)",end-start,'s')
    log.write("time of clustering cost:(part 1)" + str(end-start) + ' s\n')
    log.close()
    output = kmeans.predict(FVS)
    count  = Counter(output)
    least_index = min(count, key = count.get)            
    #print(output.shape)   
    change_map  = np.reshape(output,(new[0] - 4, new[1] - 4))
    
    return least_index, change_map
   
def find_PCAKmeans(imagepath1, imagepath2):
    start_all = time.time()
    print('Operating')
    
    type1 = imagepath1.split('.')[-1]
    type2 = imagepath2.split('.')[-1]
    image1 = imread(imagepath1,format=type1)
    image2 = imread(imagepath2,format=type2)

    if(len(image1.shape)>2):
        image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)   

    height, width = image1.shape[:2]
    old_size = (width, height)    
    size = (800, 800)
    image1 = cv2.resize(image1, size, interpolation=cv2.INTER_CUBIC).astype(np.int16)
    image2 = cv2.resize(image2, size, interpolation=cv2.INTER_CUBIC).astype(np.int16)
    new_size = np.asarray(image1.shape)

    start = time.time()
    diff_image = abs(image1 - image2)
    end = time.time()
    print('runtime of diff: ' + str(end - start) + ' seconds')
    log = open('log.txt','a')
    log.write('runtime of diff: ' + str(end - start) + ' seconds\n')
    log.close()
    #imwrite(imagepath2.split(type2)[0]+'diff.'+type1, diff_image)
    start = time.time()

    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25), 25))
    mean_vec = np.zeros((25))
    mean_vec_part = np.zeros((25))
    
    n = 12
    thr = []
    length = int(new_size[0] / (5*n) )
    for i in range(n):
        up1 = i * length * 5
        down1 = (i+1) * length * 5 if i<n-1 else new_size[0] 
        t = myThread(i,"Thread-"+str(i))
        t.getfunc(find_vector_set) 
        t.getargs(diff_image=diff_image[up1:down1,:],new_size=(length,new_size[1]))
        thr.append(t)
    
    for t in thr:   
        t.start()
        t.join()
        up2 =  t.threadID * int(length * new_size[1]/5)
        down2 = (t.threadID+1) * int(length * new_size[1]/5)
        vector_set[up2:down2,:],mean_vec_part = t.getresult()
        mean_vec = mean_vec + mean_vec_part
    mean_vec = mean_vec / n

    end = time.time()
    print("time of find_vector_set cost",end-start,'s')
    log = open('log.txt','a')
    log.write("time of find_vector_set cost" + str(end - start) + ' seconds\n')
    log.close()
    
    start = time.time()
    pca  = PCA()
    pca.fit(vector_set)
    EVS = pca.components_
    #print(EVS.shape)
    end = time.time()
    print("time of PCA cost",end-start,'s')
    log = open('log.txt','a')
    log.write("time of PCA cost" + str(end - start) + ' seconds\n')
    log.close()
    
    FVS = find_FVS(EVS, diff_image, mean_vec, new_size)    
    
    #print('\ncomputing k means')
    components = 5
    least_index, change_map = clustering(FVS, components, new_size)
    
    start = time.time()
    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
    end = time.time()
    print('runtime of highlight: ' + str(end - start) + ' seconds')
    log = open('log.txt','a')
    log.write('runtime of highlight: ' + str(end - start) + ' seconds\n')
    log.close()

    start = time.time()    
    change_map = change_map.astype(np.uint8)
    kernel     = np.ones((9,9), dtype=np.uint8)
    ChangeMap = cv2.resize(change_map, old_size, interpolation=cv2.INTER_CUBIC)
    cleanChangeMap = cv2.erode(ChangeMap,kernel)
    cleanChangeMap = cv2.dilate(cleanChangeMap,kernel)
    end = time.time()
    #print('runtime of erode&dilate: ' + str(end - start) + ' seconds')
    #print('runtime of all: ' + str(end - start_all) + ' seconds') 
    log = open('log.txt','a')
    log.write('runtime of erode&dilate: ' + str(end - start) + ' seconds\n')
    log.write('runtime of all: ' + str(end - start_all) + ' seconds\n') 
    log.close()
    imwrite(imagepath2.split(type2)[0]+"changemap."+type1, ChangeMap)
    imwrite(imagepath2.split(type2)[0]+"cleanchangemap."+type1, cleanChangeMap)
    
if __name__ == "__main__":
    a = './ElephantButte_08201991.jpg'
    b = './ElephantButte_08272011.jpg'
    a1 = './001.jpg'
    a2 = './001.001.jpg'
    b1 = './Andasol_09051987.jpg'
    b2 = './Andasol_09122013.jpg'
    tst1 = './1599621322280.jpg'
    tst2 = './1599621322244.jpg-matched.jpg'
    find_PCAKmeans(a1, a2)
           
    
