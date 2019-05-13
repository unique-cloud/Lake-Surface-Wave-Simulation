lake.py took 11.2340610027 seconds, while lake.o took 0.515487 seconds.
The substantial increase in time of lake.py may come from CPU-GPU communication overhead, as loop control happens on the CPU side; and possibly the dynamic interpretation overhead of python.

The time complexity of the program is O(N^2 * num_iter + npebs), therefore N affects the execution time the most, num_iter the second most, and npebs has little effect on execution time when it's relatively much smaller than N^2 * num_iter. Experiments do support this analysis within margin of error.
N = 512, npebs = 16, num_iter = 400: 11.2340610027 seconds
N = 256, npebs = 16, num_iter = 400: 3.87027096748 seconds
N = 512, npebs = 8, num_iter = 400: 11.3602399826 seconds
N = 512, npebs = 16, num_iter = 200: 5.6457130909 seconds

lake-horo.py took 149.932034969 seconds, and lake.py took 11.3402149677 seconds.
The massive increase in time comes from MPI_Bcast overhead in every loop iteration, even that we have mitigated the amount of data being transfered to just 3 * N for 13-point stencil.