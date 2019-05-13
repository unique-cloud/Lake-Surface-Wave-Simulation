# heatmap for lake.cu

set terminal png

set xrange[0:1]
set yrange[0:1]
set cbrange[-0.4:0.4]

set output 'lake_c_0.png'
plot 'lake_c_0.dat' using 1:2:3 with image
set output 'lake_c_1.png'
plot 'lake_c_1.dat' using 1:2:3 with image
