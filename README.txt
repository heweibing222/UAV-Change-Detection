1. Python environment installation. Install CONDA on the client (Linux Environment) and 
server (Windows Environment) respectively, with its own Python environment (version 3.6 or above).

2. Java environment installation. Directly install JDK on the client, install JDK on the server,
 and configure environment variables.
 
3. C + + environment installation. Install gcc, g++, cmake on the client.

4. OpenCV environment installation. Install OpenCV (version 3.4.0) and opencv_contrib on the client.

5. Install Python libraries. Install Python libraries on the client and server through CONDA, 
including numpy, skimage, jpype, imageio, matplotlib and CV2.

6. On the client side, use ¡°cmake install¡± to install the shooting control module.

7. First, on the server side, use the command line ¡°python3 main.py -s server -t [phase identification]¡± to start.

8. Then on the client side, use the command line ¡°python3 main.py -s client -t [phase identification] 
-ip [IP address] -d [GPS acquisition delay] -l [distance determination threshold] -n [number of acquired images]
 -m [acquisition mode]¡± to start.
 
9. Design UAV route, open the wireless connection and collect data in different time phases.
