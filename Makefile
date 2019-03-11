.POSIX:
CXX      = g++
CXXFLAGS = -I. -Wall -O2 -fopenmp
CC       = gcc
CFLAGS   = -std=c99 -Wall -O2
LDLIBS   = -lm

# Add .exe on Windows
EXEEXT   = 

all: embed$(EXEEXT) draw$(EXEEXT) mtx2csr$(EXEEXT)

embed$(EXEEXT): spectralDrawing.cpp
	$(CXX) $(CXXFLAGS) -std=c++11 -Wno-attributes spectralDrawing.cpp -o embed$(EXEEXT)
mtx2csr$(EXEEXT): mtx2csr.cpp
	$(CXX) $(CXXFLAGS) mtx2csr.cpp -o mtx2csr$(EXEEXT)
draw$(EXEEXT): draw_graph.c
	$(CC) $(CFLAGS) draw_graph.c lodepng.c -o draw$(EXEEXT) $(LDLIBS)
clean:
	rm -f embed$(EXEEXT) mtx2csr$(EXEEXT) draw$(EXEEXT)
