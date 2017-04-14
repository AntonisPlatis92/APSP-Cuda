# APSP Cuda

3 different implementations (Cuda1, Cuda2 and Cuda3) of the APSP Algorithm in Cuda.

In 1-Chandler-18.304lecture1.pdf you can find info about the algorithm.

Serial.c is the serial algorithm written in C.

-In Cuda1 each thread of the Cuda Grid is calculating one cell value
-In Cuda2 each thread of the Cuda Grid is calculating one cell value, while using a shared block table
-In Cuda3 each thread of the Cuda Grid is calculating 4 cell value, while using a shared block table

The code was written in CUDA (.cu)
