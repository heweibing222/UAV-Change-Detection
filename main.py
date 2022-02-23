'''
Cpoyright(C) 2020 Peng Shuying
This code is running on client(Ubuntu on TX2).  The program will read the GPS parameters, 
such as the acquisition interval, task status, the distance threshold for judging whether the 
image is the same place or not (compared with the previous epoch), IP address, and other 
key parameters. For the first flight mission, we only need to let the UAV operate according 
to the preset route, and then let the system accord to a certain time interval, trigger the 
imaging task while obtaining the GPS information of the current waypoint, and then send 
the images back to the ground station. For the follow-up flight mission, in addition to 
taking images according to the same flight route, the captured image and the image from 
the first flight mission should be registered first, and other tasks like change detection, 
real-time image processing, etc., are realized by the airborne development board. It use 
Python process pool mechanism to automatically realize process level parallel acceleration.
'''
import os
import numpy as np
import argparse
from imageio import imread,imwrite
from skimage.transform import rescale
from skimage.color import rgb2gray
import time
from jpype import *
import concurrent
from threading import Thread,Lock

import imageMatching as im
import PCAKmeansThread as pcak

img1_info = []    # [[name,x,y,]]
img2_info = []

def dealing(dir1='t_0', dir2='t_0', limit = 70):

    if dir1 == dir2:
        return
    try:
        gps1 = open(dir1+'/pos.txt', 'r')
        gps2 = open(dir2+'/pos.txt', 'r')
    except:
        print("Failed to open GPS file.")
    else: 
        r1 = gps1.readline().split(',')
        img1_info.append(r1)
        while(len(r1)!=1):
            r1 = gps1.readline().split(',')
            img1_info.append(r1)
        r2 = gps2.readline().split(',')
        img2_info.append(r2)
        while(len(r2)!=1):   
            r2 = gps2.readline().split(',')
            img2_info.append(r2)
        j = 0
        for i in range(len(img1_info)-1):
            Dis = 10000
            while(j < len(img2_info)-1):
                dx = int(img1_info[i][1]) - int(img2_info[j][1])
                dy = int(img1_info[i][2]) - int(img2_info[j][2])
                if(dx**2+dy**2 < Dis):
                    Dis = dx**2+dy**2
                    j = j + 1
                else:
                    break
            j = j - 1
            
            if (j<0):
                break
            if(Dis <= int(limit)):
                log = open('log.txt','a')
                log.write('^^^^^^^^^^match:'+dir1+'/'+img1_info[i][0]+','+dir2+'/'+img2_info[j][0]+'^^^^^^^^^\n')
                log.close
                print('\nmatch:'+dir1+'/'+img1_info[i][0]+','+dir2+'/'+img2_info[j][0])
                mat1 = im.image_matching(dir1+'/'+img1_info[i][0],dir2+'/'+img2_info[j][0])   
                pcak.find_PCAKmeans(dir1+'/'+img1_info[i][0],mat1)
    
    return

def trans_client(ipaddr = '192.77.108.240', from_path = 't_0',\
                 port=9000, jar_path = './uav-client.jar'):
    '''
    order = 'java -jar uav-client.jar'
    order = order + ' ' + str(ipaddr)
    order = order + ' ' + str(port)
    order = order + ' ' + str(from_path)
    '''
    #jar_path = os.path.join(os.path.abspath('.'), r'/home/yjchen/workdir/v2/uav-client.jar')
    jvm_path = getDefaultJVMPath()
    startJVM(jvm_path,'-Djava.class.path={}'.format(jar_path))
    jp = JPackage('com.yjchen.uav.client')
    #print(str(ipaddr)+' '+str(port)+' '+str(from_path))
    jc = jp.ClientApplication()
    x = []
    x.append('-h')
    x.append(str(ipaddr))
    x.append('-p')
    x.append(str(port))
    x.append('-f')
    x.append(str(from_path))
    print(x)
    jc.main(x)
    
    shutdownJVM()
   
def trans_server(port = 9000,\
                 jar_path = './uav-server.jar', \
                 to_path = 't_0'):
    #os.system('D:')
    #os.system('cd D:\C508\change-Detection-in-Satellite-Imagery-master-psy\bin')
    order = 'java -jar uav-server.jar'
    order = order + ' ' + str(port)
    order = order + ' ' + str(to_path)
    os.system(order)
    '''jvm_path = getDefaultJVMPath()
    startJVM(jvm_path,'-Djava.class.path={}'.format(jar_path))
    jp = JPackage('com.yjchen.uav.server')
    #print(str(port)+' '+str(to_path))
    js = jp.ServerApplication()
    x = []
    x.append(str(port))
    x.append(str(to_path))
    js.main(x)
    shutdownJVM() '''

def main():
    dir1 = 't_0'
    dir2 = 't_'
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', required=True,\
                        help="System as server/client. --Must needed.")  
    parser.add_argument('-t', '--times', required=True, type=str,\
                        help="times of task. --Must needed.")  
    parser.add_argument('-ip', '--ipaddr', default='192.77.108.240',\
                        help="The target ip addr. --Must needed.")
    parser.add_argument('-d', '--delay', default=3, type=int,\
                        help="Delay of GPS checking(with s).")
    parser.add_argument('-l', '--limit', default=100, type=int,\
                        help="Distance limitation of GPS location.")
    parser.add_argument('-n', '--num', default=5, type=int,\
                        help="Count of photos needed.")
    parser.add_argument('-x', '--width', default=1600, type=int,\
                        help="Width of photos.")
    parser.add_argument('-y', '--height', default=900, type=int,\
                        help="Height of photos.")
    parser.add_argument('-m', '--mode', default=1, type=int,\
                        help="mode of collecting data:(1)certain delay;(2)distance judging;(3)button trigger.")
    args = parser.parse_args()
    
    dir2 = dir2 + args.times
    futures = set()
    if os.path.isdir(dir2) == 0:
        os.makedirs(dir2)      
    if args.system == 'server':
        print('Operating...\n')
        trans_server(to_path = dir2)     
    elif args.system == 'client':
        print('Operating...\n')
        '''
        import camera_v1 as cam
        cam.start(str(dir2), args.delay, args.num,\
                  args.width, args.height, args.mode)
                  '''
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures.add(executor.submit(dealing,dir1,dir2, args.limit))
            #futures.add(executor.submit(trans_client,args.ipaddr, dir2))
        try:
            for future in concurrent.futures.as_completed(futures):
                err = future.exception()
                if err is not None:
                    raise err
        except KeyboardInterrupt:
            print("stopped by hand.")

if __name__ == "__main__":
    main()
    
'''
Test for jpype:
if True:
	from jpype import *
	jvm_path = getDefaultJVMPath()
	startJVM(jvm_path)
	java.lang.System.out.println('Success')
	shutdownJVM()

Test for trans_server():
python3 main.py -s server -t 0
Test for trans_server():
python3 main.py -s client -t 0 -ip 192.77.108.240 -d 2 -l 50 -n 0 -m 1
                                                -d 2 -l 50 -n 5 -m 2
                                    192.168.199.190
'''
'''  Testing Code
import time
import threading
def Thread_1() -> None:
    for k in range(3):
        print('Thread 1: ', k)
        time.sleep(1)
def Thread_2() -> None:
    for k in range(3):
        print('Thread 2: ', k)
        time.sleep(1)
threading.Thread(target=Thread_1).start()
time.sleep(0.5)
threading.Thread(target=Thread_2).start()
'''
