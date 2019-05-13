CC:=gcc
CCFLAGS:=-O3 -Wno-incompatible-pointer-types -Iperm-je-0.9.7/include
LDFLAGS:=-Lperm-je-0.9.7/lib -ljemalloc -lm

all: lake

lake: lake-perm.c
	$(CC) $(CCFLAGS) $(LDFLAGS) $< -o lake-perm

clean: 
	rm -f lake-perm lake_*.dat lake_*.png
