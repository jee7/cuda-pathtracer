#include <stdio.h>
#include <iostream>

__device__ const char *STR = "HELLO WORLD!";
const char STR_LENGTH = 12;

__device__ float innerProduct(float* u, float* v, int uIndex, int vIndex) {

	return u[uIndex] * v[vIndex] + u[uIndex+1] * v[vIndex+1] + u[uIndex+2] * v[vIndex+2];
}

__device__ void crossProduct(float* u, float* v, int uIndex, int vIndex, float* result) {
	result[0] = v[vIndex + 1] * u[uIndex + 2] - v[vIndex + 2] * u[uIndex + 1]; //v X v2
        result[1] = v[vIndex + 2] * u[uIndex + 0] - v[vIndex + 0] * u[uIndex + 2];
        result[2] = v[vIndex + 0] * u[uIndex + 1] - v[vIndex + 1] * u[uIndex + 0];
}

__global__ void tracer(float* field, float* triangles) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int xIndex = index % 32 - 16;
	int yIndex = 16 - (int)(index / 32);


	float p[3] = {xIndex, yIndex, 1.0};
	float v[3] = {0.0, 0.0, -1.0};

	float v1[3], v2[3], cross[3], s[3]; //Triangle vectors
	float a, f, b, c, t;

	//printf("Triangle: [%f, %f, %f], [%f, %f, %f], [%f, %f, %f]\n", triangles[0], triangles[1], triangles[2], triangles[3], triangles[4], triangles[5], triangles[6], triangles[7], triangles[8]);
	//printf("Ray: [%f, %f, %f] -> [%f, %f, %f]\n", p[0], p[1], p[2], v[0], v[1], v[2]);


	for (int i = 0; i < 1; i += 9) {
		/*
		v1[0] = triangles[i + 0 + 0] - triangles[i + 3 + 0];
		v1[1] = triangles[i + 0 + 1] - triangles[i + 3 + 1];
		v1[2] = triangles[i + 0 + 2] - triangles[i + 3 + 2];
		v2[0] = triangles[i + 6 + 0] - triangles[i + 3 + 0];
		v2[1] = triangles[i + 6 + 1] - triangles[i + 3 + 1];
		v2[2] = triangles[i + 6 + 2] - triangles[i + 3 + 2];
		*/
		v1[0] = triangles[i + 0 + 0] - triangles[i + 3 + 0];
                v1[1] = triangles[i + 0 + 1] - triangles[i + 3 + 1];
                v1[2] = triangles[i + 0 + 2] - triangles[i + 3 + 2];
                v2[0] = triangles[i + 0 + 0] - triangles[i + 6 + 0];
                v2[1] = triangles[i + 0 + 1] - triangles[i + 6 + 1];
                v2[2] = triangles[i + 0 + 2] - triangles[i + 6 + 2];

		crossProduct(v, v2, 0, 0, cross); //v X v2

		a = innerProduct(v1, cross, 0, 0);
		
		if (a > -0.00001 && a < 0.00001) {
			field[index] = 0.1;
		} else {
			f = 1.0 / a;
			s[0] = triangles[i + 0 + 0] - p[0];
			s[1] = triangles[i + 0 + 1] - p[1];
			s[2] = triangles[i + 0 + 2] - p[2];

			b = f * innerProduct(s, cross, 0, 0);

			if (b < 0.0 || b > 1.0) {
				field[index] = 0.2;
			} else {

				crossProduct(s, v1, 0, 0, cross);
				c = f * innerProduct(v, cross, 0, 0);

				if (c < 0.0 || b + c > 1.0) {
					field[index] = 0.3;
				} else {

					t = f * innerProduct(v2, cross, 0, 0);
					if (t > 0.00001) {
						field[index] = 0.4;
					} else {
						field[index] = 1.1;
						printf("Ray: [%f, %f, %f] -> [%f, %f, %f]\n", p[0], p[1], p[2], v[0], v[1], v[2]);
					}
				}
			}
		}
	}

	
	//printf("[%d, %d] ", xIndex, yIndex);
	//field[index] = 1.0;
}
__global__ void hello() {
	printf("bb");
}

int main(void)
{
	int width = 32;
	int height = 32;

	const int trianglesCount = 1;
	/*float triangles[trianglesCount][3][3] = {
		{{-5.0, -5.0, 0.0}, {5.0, -5.0, 0.0}, {0.0, 5.0, 0.0}},
	};*/
	float triangles[9] = {
		-10.0, -10.0, 0.0, 
		10.0, -10.0, 0.0, 
		0.0, 10.0, 0.0
	};

	float near = 0.0;
	float far = 100.0;
	float left = width / -2.0;
	float right = width / 2.0;
	float top = height / 2.0;
	float bottom = height / -2.0;

	//To normalized device coordinates
	/*
	for (int i = 0; i < trianglesCount; i++) {
		for (int v = 0; v < 3; v++) {
			triangles[i][v][0] = triangles[i][v][0]	/ right;
			triangles[i][v][1] = triangles[i][v][1] / top;
			triangles[i][v][2] = triangles[i][v][2] / far;
		}
	}*/

	float result[width][height];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			result[i][j] = 0.0;
		}
	}

	int num_bytes = width * height * sizeof(float);
	int num_threads = width * height;
	int num_blocks = 1;

	float *host_array = 0;
	host_array = (float*)malloc(num_bytes);

	// cudaMalloc a device array
	float *device_array = 0;
	float *device_triangles = 0;
	cudaMalloc((void**)&device_array, num_bytes);

	float triangles_num_bytes = 9 * sizeof(float);
	cudaMalloc((void**)&device_triangles, triangles_num_bytes);

	cudaMemcpy(device_triangles, triangles, triangles_num_bytes, cudaMemcpyHostToDevice);


	//tracer<<<1,1>>>(device_array, device_triangles);
	tracer<<<num_blocks, num_threads>>>(device_array, device_triangles);
	cudaDeviceSynchronize();

	// download and inspect the result on the host:
	cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);


	for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
			//std::cout << result[i][j] << ", ";
			std::cout << host_array[i * width + j] << " ";
                }
		std::cout << std::endl;
        }

	return 0;
}

