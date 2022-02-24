#ifndef CAM_GPS_H_INCLUDED
#define CAM_GPS_H_INCLUDED

#include <stdio.h>
#include <unistd.h>
#include <ctime>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <fstream>
#include <termios.h>

#include <python3.6/Python.h>
#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <sys/stat.h>

using namespace cv;
using namespace std;

int* GetGpsData(char* buff, int gpsdata[]);
extern "C"  int* GetBuff(int gpsdata[]);
int set_serial(int fd);

bool img_trigger(int** gpsdata, int r);
int start(char* dirName, int delay, int num, double w, double h, int m);
extern "C" PyObject* WrappStart(PyObject* self, PyObject* args);
extern "C" PyMODINIT_FUNC PyInit_camera_v0(void);
extern "C" PyMODINIT_FUNC PyInit_camera_v1(void);

#endif // CAM_GPS_H_INCLUDED
