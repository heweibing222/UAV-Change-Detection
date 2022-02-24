/******  Cpoyright(C) 2020 Peng Shuying
This code is running on client(Ubuntu on TX2). It is used to collect GPS
information from GPS sensor, and convert the data format. 
******/
#include "cam_gps.h"

int Flag = 0;
//double gpsdata[3] = {0.0};
int set_serial(int fd)
{
    struct termios newttys1, oldttys1;
    /* Keep the original serial port configuration */
    if (tcgetattr(fd, &oldttys1) != 0)
    {
        perror("Setupserial 1");
        return -1;
    }
    memset(&newttys1, 0, sizeof(newttys1));
	/* Start serial data reception with CREAD and local connection mode with CLOCAL */
    newttys1.c_cflag |= (CLOCAL | CREAD); 
    newttys1.c_cflag &= ~CSIZE;
    /* Data bit selection */
        newttys1.c_cflag |= CS8;
    /* Set parity bit */
        newttys1.c_cflag &= ~PARENB;
    /*  set baud rate  */
        cfsetispeed(&newttys1, B9600);
        cfsetospeed(&newttys1, B9600);
    /* Set stop bit */
        newttys1.c_cflag &= ~CSTOPB;
    /* Set the minimum characters and waiting time.*/
    newttys1.c_cc[VTIME] = 0;
    newttys1.c_cc[VMIN] = 0; 
    tcflush(fd, TCIFLUSH);
    /* Activate the configuration for it to take effect */
    if ((tcsetattr(fd, TCSANOW, &newttys1)) != 0)
    {
        perror("com set error");
        exit(1);
    }

    return 0;
}

int* GetGpsData(char* buff, int gpsdata[])
{
	int flag = 9;
	char* result = NULL;
	result = strtok(buff, ",");
			while (flag--) {
				if (flag == 6) {
					if (!strcmp(result, "A")) Flag = 1;
					else Flag = -1;
					gpsdata[0] = Flag;
				}
                if (flag == 5) {

					double tmp = atof(result);
					gpsdata[1] = floor(tmp / 100) * 100000 + (tmp - floor(tmp / 100) * 100) / 60 * 100000;  // ½« 1¡ãÀ­Éì³É100km
				}
				if (flag == 3) {
					double tmp = atof(result);
					gpsdata[2] = floor(tmp / 100) * 100000 + (tmp - floor(tmp / 100) * 100) / 60 * 100000;
					break;
				}
				result = strtok(NULL, ",");
			}
	return gpsdata;
}

int* GetBuff(int gpsdata[])
{
	int fd = 0;
	int n = 0;
	char buff[1024];
	char* dev_name = "/dev/ttyTHS2";

	if ((fd = open(dev_name, O_RDWR | O_NOCTTY | O_NDELAY)) < 0)
	{
		perror("Can't Open the ttyUSB0 Serial Port");
		return gpsdata;
	}
	set_serial(fd);

	while (Flag == 0)
	{
		sleep(1);
		read(fd,buff,sizeof(buff));
        if(buff[0] == '$'&& buff[5] == 'C' )
            gpsdata =GetGpsData(buff, gpsdata);
	}

	close(fd);
	Flag = 0;
	return gpsdata;
}
