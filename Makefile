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