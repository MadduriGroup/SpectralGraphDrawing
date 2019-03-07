# Spectral Graph Drawing

[![Build Status](https://travis-ci.com/kmadduri/SpectralGraphDrawing.svg?branch=master)](https://travis-ci.com/kmadduri/SpectralGraphDrawing)  [![Codacy Badge](https://api.codacy.com/project/badge/Grade/1296e6349fdf46baa9b8b0fadbb51a35)](https://app.codacy.com/app/kamesh.madduri/SpectralGraphDrawing?utm_source=github.com&utm_medium=referral&utm_content=kmadduri/SpectralGraphDrawing&utm_campaign=Badge_Grade_Settings)

<img src="barth5_103.gif" style="max-width:100%"/>

This repository includes code for our 2018 Graph Algorithm Building Blocks (GABB) workshop paper [Spectral Graph Drawing: Building Blocks and Performance Analysis](https://doi.org/10.1109/IPDPSW.2018.00053). Our approaches are based on spectral graph drawing algorithms developed by Yehuda Koren, described in this [paper](https://doi.org/10.1016/j.camwa.2004.08.015). 

## Getting Started

The C++ linear algebra library [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) and the [LodePNG](https://lodev.org/lodepng/) PNG image encoder and decoder are required to compile the code. Run the `bootstrap.sh` script to get the required files from Eigen and LodePNG. We used Eigen version 3.3.2 for the results in the paper, but the latest version also works fine. Note that bootstrap.sh uses the commandline utilities curl, tar, mv, and rm. If you do not have these utilities, then please get the files using git or wget or your web browser. 

A Makefile is included for building spectralDrawing.cpp (the main file with the algorithms) and two other standalone utilities (mtx2csr.cpp for converting a Matrix Market file to an intermediate binary format; graph_draw.c for taking vertex coordinates and drawing graph edges). We tested the code on Linux (Manjaro, gcc 8.2.1) and Windows 10 + MinGW (gcc 7.2.0). 

To summarize, just doing  
`./bootstrap.sh`  
`make`  
should work on most environments.

If you are unable to run the bootstrap.sh script or get some errors, please execute the following commands to get Eigen and LodePNG.    
`curl -O https://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz`  
`tar -zvxf 3.3.7.tar.gz`  
`mv eigen-eigen-*/Eigen .`  
`rm -rf eigen-eigen-* 3.3.7.tar.gz`  
`curl -O https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.h`  
`curl -O https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.cpp`  
`mv lodepng.cpp lodepng.c`  
Our code will only use some header files from the Eigen directory. Note that Eigen generates a lot of warnings on Windows 10 + MinGW.

To build the three required executables without using `make` (or `mingw32-make.exe` on Windows), execute the following commands:  
`g++ -std=c++11 -I. -Wall -O2 -fopenmp -o embed spectralDrawing.cpp`  
`g++ -I. -Wall -O2 -o mtx2csr mtx2csr.cpp`  
`gcc -std=c99 -Wall -O2 -o draw draw_graph.c lodepng.c`  

## Usage

Get a sparse matrix in Matrix Market format. The [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) is a great resource.  
`curl -O https://sparse.tamu.edu/MM/Pothen/barth5.tar.gz`  
`tar -zvxf barth5.tar.gz`  

Convert the file to intermediate binary representation.  
`./mtx2csr barth5/barth5.mtx barth5/barth5`  

Execute the graph drawing code, choosing the desired variant. In the example below, we choose coarsening, no High Dimensional Embedding, and Koren+Tutte refinement.  
`./embed barth5/barth5.csr 1 0 3`  

Create a PNG file.  
`./draw barth5/barth5.csr barth5/barth5.csr_c1_h0_r3_eps0.nxyz barth5/barth5_c1_h0_r3.png`  

The included script run_all.sh automates this process.  

For small graphs (< 10,000 vertices), SVG images look much nicer. Example SVG and PNG files with drawings of the barth5 graph are included in this repository. A short video barth5.mp4 shows intermediate steps in creating the drawing.

## Citing this work

Please cite our paper.  
Shad Kirmani and Kamesh Madduri, "Spectral Graph Drawing: Building Blocks and Performance Analysis," in Proc. Workshop on Graph Algorithms Building Blocks (GABB), May 2018. <https://doi.org/10.1109/IPDPSW.2018.00053>.
