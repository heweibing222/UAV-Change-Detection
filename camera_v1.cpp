/******  Cpoyright(C) 2021 Peng Shuying
This code is running on client(Ubuntu on TX2). It calls and manages the camera 
module and GPS sensor independently. Only when the GPS module can connect to 
the satellite normally and output POS information is the current image stream 
stored as an image file. The program provides three image acquisition methods 
for this experiment: (1) timing capture,(2) timing acquisition combined with 
GPS positioning discrimination, and (3) manual control of capture. 
******/

#include "cam_gps.h"
using namespace cv;
using namespace std;
string imgName = "000.jpg";
string posName;
ofstream filePos;
int** gpsdata = NULL;
bool img_trigger(int** gpsdata, int r){
    int ans = 0;
    for(int i=1;i<r;i++){
        ans += abs(gpsdata[0][1]-gpsdata[i][1]);
        ans += abs(gpsdata[0][2]-gpsdata[i][2]);
    }
    return ans<=3*r ;
}

int start(char* dir, int delay, int num, double w, double h, int mode){
    /** \brief calls and manages the camera module and GPS sensor independently, and get datasets.
        * \param[in] dir the path where data is stored.
        * \param[in] delay the delay(seconds) of GPS-checking.
        * \param[in] num the max count of images.
        * \param[in] w the width of images.
        * \param[in] h the height of images.
        * \param[in] mode the mode of collecting data:(1) for certain delay;(2) for distance judging;(3) for button trigger.
        */
	string dirName = dir;
	string savePath;
	char keyCode;
	const int r = 3;
    VideoCapture cap("v4l2src device=/dev/video1 ! videoconvert ! video/x-raw ! appsink drop=true");
    if (!cap.isOpened())
    {
        cout << "Failed to open camera." << endl;
        return -1;
    } else {
		cout << "open camera." << endl;
		posName = dirName + "/pos.txt";
		filePos.open(posName, ios_base::out);
        filePos.close();    //init gps files.
	}
	gpsdata = (int**)malloc(r * sizeof(int*));
	for(int i=0;i<r;i++){
        gpsdata[i] = (int*)malloc(3 * sizeof(int));
        memset(gpsdata[i],0,3 * sizeof(int));
	}
	while(1){
        cout << "Please input \'b\' to begin program when UAV is ready." << endl;
        cin >> keyCode;
        if(keyCode == 'b')
            break;
	}
    filePos.open(posName, ios::app);
    cout << "OK. Let's start." << endl;
    cout << "Please print \'s\' or autonomally to take photos, and \'e\' to finish." << endl;
    for(int i=0,a=0;i<num;)
    {
        Mat frame;
        //imshow("original", frame);
        usleep(1e6*delay);	// delay X s
        keyCode = waitKey(30);
        if(mode == 3){
            cin >> keyCode;
            if(keyCode == 'e')  break;
            else if(keyCode == 's');
            else continue;
        }
        cap >> frame;
        gpsdata[a] = GetBuff(gpsdata[a]);
        if( gpsdata[a][0] == 1 && (mode!=2 || (img_trigger(gpsdata,r)&&mode==2)) ){
            savePath = dirName + "/";
            savePath.append(imgName);
            cout << savePath << " is saved."<<endl; 
            filePos <<imgName<<','<<gpsdata[a][1]<<','<<gpsdata[a][2]<<','<< "\n";

            imwrite(savePath, frame);    // save img
            frame.release();
            memset(gpsdata[a],0,3 * sizeof(int));
            i++;
            if(imgName.at(2)<'9')		imgName.at(2)++;
            else{
                imgName.at(2)='0';
                if(imgName.at(1)<'9')	imgName.at(1)++;
                else{
                    imgName.at(1)='0';
                    if(imgName.at(0)<'9')	imgName.at(0)++;
                    else	imgName.at(0)='0';
                }
            }
        }
        a=(a+1)%r;
    }
    filePos.close();
    return 0;
}

PyObject* WrappStart(PyObject* self, PyObject* args)
{
    char* dirName;
	int delay, num, m;
	double w, h;
 	if (!PyArg_ParseTuple(args, "siiddi", &dirName, &delay, &num, &w, &h, &m))
  		return NULL;
    return Py_BuildValue("i", start(dirName,delay,num,w,h,m));
}
static PyMethodDef test_methods[] = {
    {"start", WrappStart, METH_VARARGS, "something"},
    {NULL, NULL}
};
static PyModuleDef test_module{
    PyModuleDef_HEAD_INIT,
    "test_module",
    "the module of camera.",
    -1,
    test_methods
};

PyMODINIT_FUNC PyInit_camera_v1(void)
{
    return PyModule_Create(&test_module);
}

// Use Makefile
/*
DIR	= /usr
LIBDIR      = $(DIR)/lib/ 
BINDIR      = $(DIR)/bin/
SRCDIR      = $(DIR)/src/
HEADPATH    = $(DIR)/include/ -I ../include/

CC = gcc  -D_FILE_OFFSET_BITS=64
GXX = g++ -D_FILE_OFFSET_BITS=64
MPI_CC  =mpicc  -D_FILE_OFFSET_BITS=64
MPI_XX  =mpicxx -D_FILE_OFFSET_BITS=64

RM      =rm -fr
CP      =cp -fr
AR      =ar -r

camera_v1: camera_v1.o
	$(GXX) -fPIC -shared camera_v1.cpp getgpsdata.cpp -I $(HEADPATH) `pkg-config --cflags --libs opencv` -o camera_v1.so -lm
*/