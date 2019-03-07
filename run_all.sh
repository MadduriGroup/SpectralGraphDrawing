#!/bin/sh

if [ -z "$2" ]; then
	echo "Usage: ./run.sh <SuiteSparse group name> <mat-name>"
	exit 1
fi
matgroup=$1
mat=$2

mtx=data/$mat/$mat.mtx
if [ ! -f $mtx ]; then
	echo $mtx 'does not exist'
	echo 'getting from SuiteSparse matrix repository'
	cd data
	curl -O https://sparse.tamu.edu/MM/$matgroup/$mat.tar.gz
	tar -zvxf $mat.tar.gz
	cd ..
fi

if [ ! -f data/$mat/yifan.gif ]; then
	curl -O http://yifanhu.net/GALLERY/GRAPHS/GIF_SMALL/${matgroup}@${mat}.gif 
	mv ${matgroup}@${mat}.gif data/$mat/yifan.gif
fi

#Convert mtx file to csr format
./mtx2csr data/$mat/$mat.mtx data/$mat/$mat

#Generate coordinates
for c in 0
do
	for h in 0 1
	do
		for r in 0 1 2 3
		do
			if [ ! -f data/$mat/c${c}_h${h}_r${r}.png ]; then
				./embed data/$mat/$mat.csr $c $h $r
				mv data/$mat/$mat.csr_* data/$mat/coord.mtx
				cp data/$mat/coord.mtx data/$mat/coord_${c}_${h}_${r}.mtx
				./draw data/$mat/$mat.csr data/$mat/coord.mtx graph_draw.png
				cp graph_draw.png data/$mat/c${c}_h${h}_r${r}.png
        			rm graph_draw.png
				rm data/$mat/coord.mtx
			fi
		done
	done
done	


for c in 1
do
	for h in 0
	do
		for r in 0 1 2 3
		do
			if [ ! -f data/$mat/c${c}_h${h}_r${r}.png ]; then
				./embed data/$mat/$mat.csr $c $h $r
				mv data/$mat/$mat.csr_* data/$mat/coord.mtx
				cp data/$mat/coord.mtx data/$mat/coord_${c}_${h}_${r}.mtx
				./draw data/$mat/$mat.csr data/$mat/coord.mtx graph_draw.png
				cp graph_draw.png data/$mat/c${c}_h${h}_r${r}.png
        			rm graph_draw.png
				rm data/$mat/coord.mtx
			fi
		done
	done
done	


for c in 2
do
	for h in 0
	do
		for r in 1
		do
			if [ ! -f data/$mat/coarse.png ]; then
				./embed data/$mat/$mat.csr $c $h $r
				mv data/$mat/$mat.csr_* data/$mat/coord_coarse.mtx
				mv graph_coarse.csr data/$mat/.
				mv graph_coarse.mtx data/$mat/.
				./draw data/$mat/$mat.csr data/$mat/coord.mtx graph_draw.png
				cp graph_draw.png data/$mat/coarse.png
        			rm graph_draw.png
			fi
		done
	done
done	

gf=data/$mat/graphs.html
if [ ! -f $gf ]; then
	echo "<!DOCTYPE html>" >> $gf
	echo "<html>" >> $gf
	echo "<title>"$mat"</title>" >> $gf
	echo '<xmp theme="united" style="display:none;">' >> $gf
	# Markdown goes here
	echo "## Koren's alg  " >> $gf
	echo "![Koren](c0_h0_r1.png)" >> $gf
	echo "## Tutte's alg  " >> $gf
	echo "![Tutte](c0_h0_r2.png)" >> $gf
	echo "## Both Koren's and Tutte's alg  " >> $gf
	echo "![Koren + Tutte](c0_h0_r3.png)" >> $gf
	echo "## HDE only " >> $gf
	echo "![HDE only](c0_h1_r0.png)" >> $gf
	echo "## HDE + Koren's alg  " >> $gf
	echo "![HDE + Koren](c0_h1_r1.png)" >> $gf
	echo "## HDE + Tutte's alg  " >> $gf
	echo "![HDE + Tutte](c0_h1_r2.png)" >> $gf
	echo "## HDE + Koren's + Tutte's alg  " >> $gf
	echo "![HDE + Koren + Tutte](c0_h1_r3.png)" >> $gf
	echo "## Coarsening + Koren's alg  " >> $gf
	echo "![Coarsening + Koren](c1_h0_r1.png)" >> $gf
	echo "## Coarsening + Tutte's alg  " >> $gf
	echo "![Coarsening + Tutte](c1_h0_r2.png)" >> $gf
	echo "## Coarsening + Tutte's + Koren's alg  " >> $gf
	echo "![Coarsening + Tutte + Koren](c1_h0_r3.png)" >> $gf
	echo "## Coarse graph  " >> $gf
	echo "![Coarse](coarse.png)" >> $gf
  echo "## Yifan Hu's visualization  "  >> $gf
	echo "![Yifan](yifan.gif)" >> $gf
	echo '</xmp>' >> $gf
	echo '<script src="http://strapdownjs.com/v/0.2/strapdown.js"></script>
' >> $gf
    echo "</html>" >> $gf
fi
