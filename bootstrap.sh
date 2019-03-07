#!/bin/sh
# set up Eigen
curl -O https://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz
tar -zvxf 3.3.7.tar.gz
mv eigen-eigen-*/Eigen .
rm -rf eigen-eigen-* 3.3.7.tar.gz

# get the lode PNG encoder
curl -O https://github.com/lvandeve/lodepng/blob/master/lodepng.h
curl -O https://github.com/lvandeve/lodepng/blob/master/lodepng.cpp
mv lodepng.cpp lodepng.c
