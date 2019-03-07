# Spectral Graph Drawing

This repository includes code for our 2018 Graph Algorithm Building Blocks (GABB) workshop paper [Spectral Graph Drawing: Building Blocks and Performance Analysis](https://doi.org/10.1109/IPDPSW.2018.00053). Our approaches are based on spectral graph drawing algorithms developed by Yehuda Koren, described in this [paper](https://doi.org/10.1016/j.camwa.2004.08.015). 

## Getting Started

Please see spectralDrawing.cpp. The C++ linear algebra library [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) is required for compilation. We used Eigen version 3.3.2 for the results in the paper, but the latest version also works. Just download and copy the Eigen folder to the current working directory. A Makefile is included for building spectralDrawing.cpp and two other utilities (mtx2csr.cpp for converting a Matrix Market file to an intermediate binary format; graph_draw.c for taking vertex coordinates and creating a PNG file). We tested the code on Linux and Windows 10 + MinGW. To build the three executables without the Makefile, execute the following commands:  
`g++ -Wall -I. -O2 -o embed spectralDrawing.cpp`  
`g++ -Wall -I. -O2 -o mtx2csr mtx2csr.cpp`  
`gcc -Wall -O2 -o draw draw_graph.c lodepng.c`  
Note that Eigen generates a lot of warnings on Windows 10 + MinGW.

### Usage Examples

Get a sparse matrix in Matrix Market format. The [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) is a great resource.  
`curl -O https://sparse.tamu.edu/MM/Pothen/barth5.tar.gz`  
`tar -zvxf barth5.tar.gz`  

Convert the file to intermediate binary representation.  
`./mtx2csr barth5/barth5.mtx barth5/barth5`  

Execute the graph drawing code, choosing the desired variant. In the example below, we choose coarsening, HDE, and Koren+Tutte refinement.  
`./embed barth5/barth5.csr 1 1 3`  

Create a PNG file.  
`./draw.exe barth5/barth5.csr barth5/barth5.csr_c0_h1_r3_eps0.nxyz barth5/barth5_c0_h1_r3.png`  

The included script run_all.sh automates this process.  

For small graphs (< 10,000 vertices), SVG images look much nicer.

## Citing this work

Please cite our paper.  
Shad Kirmani and Kames Madduri, "Spectral Graph Drawing: Building Blocks and Performance Analysis," in Proc. Workshop on Graph Algorithms Building Blocks (GABB), May 2018. https://doi.org/10.1109/IPDPSW.2018.00053.
