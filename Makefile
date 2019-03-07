.POSIX:
CXX      = g++
CXXFLAGS = -I. -std=c++11 -Wall -O2 -fopenmp
CC       = gcc
CFLAGS   = -std=c99 -Wall -O2

# Add .exe on Windows
EXEEXT   = 

all: embed$(EXEEXT) draw$(EXEEXT) mtx2csr$(EXEEXT)

embed$(EXEEXT): spectralDrawing.cpp
	$(CXX) $(CXXFLAGS) -o embed$(EXEEXT) spectralDrawing.cpp
mtx2csr$(EXEEXT): mtx2csr.cpp
	$(CXX) $(CXXFLAGS) -o mtx2csr$(EXEEXT) mtx2csr.cpp
draw$(EXEEXT): draw_graph.c
	$(CC) $(CFLAGS) -o draw$(EXEEXT) draw_graph.c lodepng.c
clean:
	rm -f embed$(EXEEXT) mtx2csr$(EXEEXT) draw$(EXEEXT)
