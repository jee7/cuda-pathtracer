#define the complier
COMPILER = nvcc

# compilation settings, optimization, precision, parallelization...  
# 
FLAGS = -arch=sm_35 --relocatable-device-code=true # -rdc=true #-O2 

# libraries.. /<libdir>/libmkl_blas.[so|a] 
#LIBS =  -L/home/hpc_jee7/devil/lib/lib -lIL -lILU -lILUT #  -L/home/hpc_jee7/devil/lib/lib/ -llibILU  -L/home/hpc_jee7/devil/lib/lib -llibILUT
LIBS = -lcudadevrt -lcurand -I/usr/local/cuda/include/



# source list for main program
SOURCES = pathTracer.cu

test: $(SOURCES)
	${COMPILER} -g -o pathTracer $(FLAGS) $(SOURCES) $(LIBS) 

clean:
	rm *.o 

clobber:
	rm pathTracer
