#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ctime>

#include "cudaQueue.cu"
#include <curand.h>
#include <curand_kernel.h>

#define ITERATIONS 800
#define BOUNCES 4 //At least 3!! (this means at least 1 in reality)
#define WIDTH 256
#define HEIGHT 256
#define FIELD_SIZE WIDTH*HEIGHT

struct hit {
	bool isHit;
	int index;
	float t;
};

struct Parameters {
	int startX;
	int startY;
	int width;
	int height;
	int iterations;
	int bounces;
	unsigned long long size;
};

__device__ Queue q;

__device__ float randomFloat(float a, float b, curandState* state) {

	return (b - a) * curand_uniform(state) + a;
}

__device__ float innerProduct(float* u, float* v, int uIndex, int vIndex) {

        return u[uIndex] * v[vIndex] + u[uIndex+1] * v[vIndex+1] + u[uIndex+2] * v[vIndex+2];
}

__device__ void crossProduct(float* u, float* v, int uIndex, int vIndex, float* result) {
        result[0] = u[vIndex + 1] * v[uIndex + 2] - u[vIndex + 2] * v[uIndex + 1]; //v X v2
        result[1] = u[vIndex + 2] * v[uIndex + 0] - u[vIndex + 0] * v[uIndex + 2];
        result[2] = u[vIndex + 0] * v[uIndex + 1] - u[vIndex + 1] * v[uIndex + 0];
}


__device__ void normalize(float* v) {
        float length = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        v[0] /= length;
        v[1] /= length;
        v[2] /= length;
}


__device__ void randomReflect(float* normals, int normalIndex, float* result, curandState* state) {
	float sphere[3] = {randomFloat(-1.0, 1.0, state), randomFloat(-1.0, 1.0, state), randomFloat(-1.0, 1.0, state)};
	while (sqrt(sphere[0]*sphere[0] + sphere[1]*sphere[1] + sphere[2]*sphere[2]) > 1.0 or innerProduct(normals, sphere, normalIndex, 0) < 0.0) {
		sphere[0] = randomFloat(-1.0, 1.0, state);
		sphere[1] = randomFloat(-1.0, 1.0, state);
		sphere[2] = randomFloat(-1.0, 1.0, state);
	}
	normalize(sphere);

	result[0] = sphere[0];
	result[1] = sphere[1];
	result[2] = sphere[2];
}

__device__ void reflect(float* incident, float* normal, int incidentIndex, int normalIndex, float* result) {
        float d = innerProduct(incident, normal, incidentIndex, normalIndex);

        result[0] = incident[incidentIndex + 0] - 2.0 * d * normal[normalIndex + 0];
        result[1] = incident[incidentIndex + 1] - 2.0 * d * normal[normalIndex + 1];
        result[2] = incident[incidentIndex + 2] - 2.0 * d * normal[normalIndex + 2];
}

__device__ float checkItersection(float* rayStart, float* rayDirection, float* triangles, int triangleIndex) {
	float v1[3], v2[3], cross[3], s[3]; //Triangle vectors
        float a, f, b, c, t;

	v1[0] = triangles[triangleIndex + 3 + 0] - triangles[triangleIndex + 0 + 0];
        v1[1] = triangles[triangleIndex + 3 + 1] - triangles[triangleIndex + 0 + 1];
        v1[2] = triangles[triangleIndex + 3 + 2] - triangles[triangleIndex + 0 + 2];
        v2[0] = triangles[triangleIndex + 6 + 0] - triangles[triangleIndex + 0 + 0];
        v2[1] = triangles[triangleIndex + 6 + 1] - triangles[triangleIndex + 0 + 1];
        v2[2] = triangles[triangleIndex + 6 + 2] - triangles[triangleIndex + 0 + 2];

        crossProduct(rayDirection, v2, 0, 0, cross); //v X v2

        a = innerProduct(v1, cross, 0, 0);

        if (a > -0.00001 && a < 0.00001) {
                      
		return INFINITY;
        }
        f = 1.0 / a;
	s[0] = rayStart[0] - triangles[triangleIndex + 0 + 0];
        s[1] = rayStart[1] - triangles[triangleIndex + 0 + 1];
        s[2] = rayStart[2] - triangles[triangleIndex + 0 + 2];


        b = f * innerProduct(s, cross, 0, 0);
	if (b < 0.0 || b > 1.0) {
                                
		return INFINITY;
       	}

	crossProduct(s, v1, 0, 0, cross);
        c = f * innerProduct(rayDirection, cross, 0, 0);

        if (c < 0.0 || b + c > 1.0) {

		return INFINITY;
        }

        t = f * innerProduct(v2, cross, 0, 0);
        if (t > 0.00001) {
		//printf("Ray: [%f, %f, %f] -> [%f, %f, %f]\n", p[0], p[1], p[2], v[0], v[1], v[2]);

		return t;
        }

	return INFINITY;
}

__device__ hit castRay(float* rayStart, float* rayDirection, float* vertices, int* faces) {
	//TODO Use normals to cull some intersections?

	float intersection;
        float closestIntersection = INFINITY;
        int closestIntersectionIndex = -1;
	float triangle[9];
        for (int i = 0; i < (3 * 14); i += 3) {
		triangle[0] = vertices[3 * faces[i + 0] + 0];
		triangle[1] = vertices[3 * faces[i + 0] + 1];
		triangle[2] = vertices[3 * faces[i + 0] + 2];
		triangle[3] = vertices[3 * faces[i + 1] + 0];
		triangle[4] = vertices[3 * faces[i + 1] + 1];
		triangle[5] = vertices[3 * faces[i + 1] + 2];
		triangle[6] = vertices[3 * faces[i + 2] + 0];
		triangle[7] = vertices[3 * faces[i + 2] + 1];
		triangle[8] = vertices[3 * faces[i + 2] + 2];

                intersection = checkItersection(rayStart, rayDirection, triangle, 0);
                if (intersection < closestIntersection) {
                        closestIntersection = intersection;
                        closestIntersectionIndex = i;
                }
        }

	hit rayHit = hit();
	rayHit.isHit = (closestIntersectionIndex != -1);
	rayHit.t = closestIntersection;
	rayHit.index = closestIntersectionIndex;

	return rayHit;
}


__global__ void rayTrace(float* field, float* vertices, int* faces, float* normals, float* colors, Queue* q, curandState* state, Parameters* params) {
	//printf("pop-");
	Task task = q->pop();
	//printf("popped\n");
	if (!task.isValid) {
		//printf("Invalid task!\n");
		return;
	} else {
		//printf("Valid!\n");
	}
	float rayStart[3] = {task.data[0], task.data[1], task.data[2]};
	float rayDir[3] = {task.data[3], task.data[4], task.data[5]};
	int index = task.index;
	short depth = task.depth;
	//printf("RayTrace %f\n", task.data[6]);
	//printf("value: %f\n", task.value);

	if (depth == BOUNCES - 2) { //Getting to max depth, shoot to light source
		float lightSource[3] = {0.0, 9.9, -5.0};
		rayDir[0] = lightSource[0] - rayStart[0];
		rayDir[1] = lightSource[1] - rayStart[1];
		rayDir[2] = lightSource[2] - rayStart[2];
	}
	if (depth >= BOUNCES - 1) { //Max depth, add our value
		//printf("Finish %f\n", task.value);
		atomicAdd(&(field[index + 0]), task.value[0] / params->iterations);
		atomicAdd(&(field[index + 1]), task.value[1] / params->iterations);
		atomicAdd(&(field[index + 2]), task.value[2] / params->iterations);

		return;
        }

	hit rayHit = castRay(rayStart, rayDir, vertices, faces);
        if (rayHit.isHit) {
		bool isLightSource = rayHit.index >= 10 * 3;
                if (isLightSource && depth == 0) {
			atomicExch(&(field[index + 0]), colors[rayHit.index + 0]); //We hit the light source straight away
			atomicExch(&(field[index + 1]), colors[rayHit.index + 1]);
			atomicExch(&(field[index + 2]), colors[rayHit.index + 2]);
                } else {
			if (isLightSource) {
				atomicAdd(&(field[index + 0]), task.value[0] / params->iterations); //We hit the light, write the accumulated value
				atomicAdd(&(field[index + 1]), task.value[1] / params->iterations);
				atomicAdd(&(field[index + 2]), task.value[2] / params->iterations);
                        } else {

				//Generate random ray
				float randomRayDir[3];
				randomReflect(normals, rayHit.index, randomRayDir, state);

				float reflectance = 0.5; //Can be color vector
				float brdf = 2.0 * reflectance * innerProduct(normals, randomRayDir, rayHit.index, 0);

				rayStart[0] += rayHit.t * rayDir[0];
				rayStart[1] += rayHit.t * rayDir[1];
				rayStart[2] += rayHit.t * rayDir[2];
			
				task.data[0]   = rayStart[0];
				task.data[1]   = rayStart[1];
				task.data[2]   = rayStart[2];
				task.data[3]   = randomRayDir[0];
				task.data[4]   = randomRayDir[1];
				task.data[5]   = randomRayDir[2];
				task.depth    += 1;
				task.value[0] *= brdf * colors[rayHit.index + 0];
				task.value[1] *= brdf * colors[rayHit.index + 1];
				task.value[2] *= brdf * colors[rayHit.index + 2];

		                bool isSuccess = q->push(task);
				if (!isSuccess) {
					printf("Could not add bounce to queue! Very bad!\n");
				}
			}
                }
        } else {
		//printf("Miss\n");
        }
}


__global__ void tracer(float* field, float* vertices, int* faces, float* normals, float* colors, Queue* q, curandState* state, Parameters* params) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int xIndex = index % params->height - params->width / 2;
	int yIndex = params->height / 2 - (int)(index / params->width);
	long int rayCount = params->iterations * params->bounces;

	if (index >= params->size) { //Some threads are outside the field
		//printf("Dead %d\n", index);
		rayCount = 0;
	} else {

		curand_init(1337 + index, 0, 0, &state[index]);

		float zoom = params->width / 21.0;

		float pixelCoordinates[3] = {xIndex / zoom, yIndex / zoom, 1.0}; //Clipping box
		float rayStart[3] = {0.0, 0.0, 20.0}; //Camera position
		float rayDirection[3];
		rayDirection[0] = pixelCoordinates[0] - rayStart[0];
		rayDirection[1] = pixelCoordinates[1] - rayStart[1];
		rayDirection[2] = pixelCoordinates[2] - rayStart[2];

		for (int i = 0; i < ITERATIONS; i++) {

			Task task = Task();
			task.data[0]  = rayStart[0];
			task.data[1]  = rayStart[1];
			task.data[2]  = rayStart[2];
			task.data[3]  = rayDirection[0];
			task.data[4]  = rayDirection[1];
			task.data[5]  = rayDirection[2];
			task.index    = index * 3;
			task.depth    = 0;
			task.value[0] = 1.0;
			task.value[1] = 1.0;
			task.value[2] = 1.0;
			task.isValid = true;

			bool canPush = q->push(task);
			if (!canPush) {
				printf("Could not add initial rays! Very bad!"); return;
				//Can not resolve the rays here, because different blocks can not be synchronized
			}
		}
	}
	//Would this be really needed?
	//cudaDeviceSynchronize();
	//__syncthreads();

	if (rayCount > 0) {
		int numBlocks, numThreads;
		if (rayCount * BOUNCES < 1024) {
			numBlocks = 1;
			numThreads = rayCount * params->bounces;
		} else {
			numBlocks = int(rayCount * params->bounces / 32) + 1;
			numThreads = 32;
			
		}

		//printf("Ray count: %ld\n", rayCount);
		//printf("blocks %d, threads %d\n", numBlocks, numThreads);
	        rayTrace<<<numBlocks,numThreads>>>(field, vertices, faces, normals, colors, q, state, params);
	}

	cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
                printf("CUDA error (kernel): %s\n", cudaGetErrorString(error));
        }

	//printf("Triangle: [%f, %f, %f], [%f, %f, %f], [%f, %f, %f]\n", triangles[0], triangles[1], triangles[2], triangles[3], triangles[4], triangles[5], triangles[6], triangles[7], triangles[8]);
	//printf("Ray: [%f, %f, %f] -> [%f, %f, %f]\n", p[0], p[1], p[2], v[0], v[1], v[2]);
	//printf("[%d, %d] ", xIndex, yIndex);
	//field[index] = 1.0;
}
__global__ void hello() {
	printf("bb");
}

struct vector {
	float x;
	float y;
	float z;
};

vector makeVector(float* p1, float* p2, int p1Index, int p2Index) {
	vector v;
	v.x = p2[p2Index + 0] - p1[p1Index + 0];
	v.y = p2[p2Index + 1] - p1[p1Index + 1]; 
	v.z = p2[p2Index + 2] - p1[p1Index + 2];

	return v;
}

vector crossProduct(vector u, vector v) {
	vector result;
        result.x = u.y * v.z - u.z * v.y; 
        result.y = u.z * v.x - u.x * v.z;
        result.z = u.x * v.y - u.y * v.x;

	return result;
}

vector normalize(vector v) {
	float length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	vector result;
	result.x = v.x / length;
	result.y = v.y / length;
	result.z = v.z / length;

	return result;
}


int main(void)
{
	unsigned long long size = WIDTH * HEIGHT;

	const int trianglesCount = 14;
	const int verticesCount = 12 * 3; //This is not triangles count * 3, some triangles share
	const int facesCount = trianglesCount * 3; //Each triangle has a face

	float vertices[verticesCount] = {
		-10.0, -10.0,   22.0, //0  //Left wall
		-10.0, -10.0, -10.0, //1 
		-10.0,  10.0, -10.0, //2
		-10.0,  10.0,   22.0, //3
		 10.0, -10.0,   22.0, //4  //Right wall
                 10.0, -10.0, -10.0, //5
                 10.0,  10.0, -10.0, //6
                 10.0,  10.0,   22.0, //7
	        -5.0,   9.9,   -8.0, //8 //Light source
	        -5.0,   9.9,   -2.0, //9
		 5.0,   9.9,   -2.0, //10
                 5.0,   9.9,   -8.0  //11

	};
	int faces[facesCount] = { //x, y, z
		0, 1, 2, //Left wall
		0, 2, 3,
		1, 5, 6, //Back wall
		1, 6, 2,
		5, 7, 6, //Right wall
		5, 4, 7,
		2, 7, 3, //Top wall
		2, 6, 7,
		0, 5, 1, //Bottom wall
		0, 4, 5,
		0, 7, 3, //Front wall?
		0, 4, 7,
		10, 9, 8, //Light source
		11, 10, 8
		/*8, 9, 10, //Wrong normals light source?
		8, 10, 11*/
	};
	float normals[facesCount]; //Each face has a normal
	for (int i = 0; i < facesCount; i += 3) {
		vector v1 = makeVector(vertices, vertices, 3 * faces[i + 0], 3 * faces[i + 1]);
		vector v2 = makeVector(vertices, vertices, 3 * faces[i + 0], 3 * faces[i + 2]);
		printf("v1: [%f, %f, %f]\n", v1.x, v1.y, v1.z);
		printf("v2: [%f, %f, %f]\n", v2.x, v2.y, v2.z);

		vector cross = crossProduct(v1, v2);
		cross = normalize(cross);
		normals[i + 0] = cross.x;
		normals[i + 1] = cross.y;
		normals[i + 2] = cross.z;

		printf("Normal: [%f, %f, %f]\n", normals[i + 0], normals[i + 1], normals[i + 2]);
	}
	float colors[facesCount * 3] = { //Each face has a color
		1.0, 0.0, 0.0, //Left wall
		1.0, 0.0, 0.0,
		1.0, 1.0, 1.0,
		1.0, 1.0, 1.0,
		0.0, 1.0, 0.0, //Right wall
		0.0, 1.0, 0.0,
		1.0, 1.0, 1.0, //Bottom
		1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, //Top
		1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, //Front
		1.0, 1.0, 1.0,
		1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, //Light 
		1.0, 1.0, 1.0 
	};

	//Result field
	int num_bytes = size * 3 * sizeof(float);
	float *result_field = 0;
        result_field = (float*)malloc(num_bytes);
        for (int i = 0; i < size * 3; i++) {
        	result_field[i] = 0.0; //We start with 0.0
        }


	std::clock_t timeStart = std::clock();

	//float** host_fields = 0;
	//float** device_fields = 0;
	cudaError_t error;

	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);
	//gpuCount = 1;

	//host_fields = (float**)malloc(gpuCount * sizeof(float*));
	//device_fields = (float**)malloc(gpuCount * sizeof(float*));
	float* host_fields[gpuCount];
	float* device_fields[gpuCount];

	unsigned long long gpuFieldSize;

	for (int gpuIndex = 0; gpuIndex < gpuCount; gpuIndex++) {

		Parameters params = Parameters();
	        params.width = WIDTH / gpuCount;
	        params.height = HEIGHT;
		params.startX = gpuIndex * params.width;
		params.startY = 0;
	        params.iterations = ITERATIONS;
	        params.bounces = BOUNCES;
		params.size = params.width * params.height;
		gpuFieldSize = params.size;

		printf("Field size: %ull\n", gpuFieldSize);

		//Host field for 1 GPU
		int num_bytes = params.size * 3 * sizeof(float);

	        host_fields[gpuIndex] = (float*)malloc(num_bytes);
	        for (int i = 0; i < params.size * 3; i++) {
        	        host_fields[gpuIndex][i] = 0.0; //We start with 0.0
	        }

		cudaSetDevice(gpuIndex);
		printf("GPU %d\n", gpuIndex);

		unsigned long long int memoryCap = 3500000000; //3.5GB
		printf("q_size = %llu vs %llu\n",(unsigned long long int)(memoryCap / sizeof(Task)), params.size * params.iterations);
		printf("sizeof(Task) = %d\n", sizeof(Task));
		int q_size = min((unsigned long long int)(memoryCap / sizeof(Task)), params.size * params.bounces * params.iterations);

		unsigned long neededSize = params.size * params.iterations;
		if (q_size < neededSize) {
			printf("Queue size %d is smaller than needed size %lu!\n", q_size, neededSize);

			return 1;
		}

		Parameters* device_params = 0;
		cudaMalloc((void **)&device_params, sizeof(Parameters));
		cudaMemcpy(device_params, &params, sizeof(Parameters), cudaMemcpyHostToDevice);
		

		Queue* q = 0;
		Queue* host_q = new Queue();
		host_q->init(q_size);
		cudaMalloc((void **)&q, sizeof(Queue));
		printf("Queue size: %d\n", sizeof(Queue));
		printf("Queue lists: %llu\n", q_size * sizeof(Task));
	        cudaMemcpy(q, host_q, sizeof(Queue), cudaMemcpyHostToDevice);


		//Sick... http://stackoverflow.com/questions/16024087/copy-an-object-to-device
		Task* d_data = 0;
		cudaMalloc((void **)&d_data, q_size * sizeof(Task));
		cudaMemcpy(d_data, host_q->data, q_size * sizeof(Task), cudaMemcpyHostToDevice);
		cudaMemcpy(&(q->data), &d_data, sizeof(Task *), cudaMemcpyHostToDevice);

		int num_threads, num_blocks, total_threads;
		if (params.size >= 1024) {
			num_threads = 1024;
			num_blocks = (params.size / 1024) + 1;
		} else if (params.size >= 16) {
			num_threads = 16;
			num_blocks = (params.size / 16) + 1;
		} else {
			num_threads = params.size;
			num_blocks = 1;
		}
		total_threads = num_blocks * num_threads;
		printf("Blocks: %d, ThreadPerBlock: %d\n", num_blocks, num_threads);

		/*
			Threads in different blocks cannot
			synchronize -> CUDA runtime system
			can execute blocks in any order
		*/


		// cudaMalloc a device array
		float *device_vertices = 0;
		int *device_faces = 0;
		float *device_normals = 0;
		float *device_colors = 0;

		printf("CUDA MALLOC\n");
		device_fields[gpuIndex] = 0;
		error = cudaMalloc((void**)&(device_fields[gpuIndex]), num_bytes);
		if (error != cudaSuccess) {
                	printf("CUDA error: %s\n", cudaGetErrorString(error));
	        }
		printf("Memory allocated to device pointer: %d\n", device_fields[gpuIndex]);

		

		float faces_num_bytes    = facesCount    * sizeof(int);
		float vertices_num_bytes = verticesCount * sizeof(float);
		float normals_num_bytes  = facesCount    * sizeof(float);
		float colors_num_bytes   = facesCount    * sizeof(float);
		cudaMalloc((void**)&device_vertices, vertices_num_bytes);
		cudaMalloc((void**)&device_faces,    faces_num_bytes);
		cudaMalloc((void**)&device_normals,  normals_num_bytes);
		cudaMalloc((void**)&device_colors,   colors_num_bytes);

		cudaMemcpy(device_faces, faces, faces_num_bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(device_vertices, vertices, vertices_num_bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(device_normals, normals, normals_num_bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(device_colors, colors, colors_num_bytes, cudaMemcpyHostToDevice);

		printf("CUDA MEMCPY\n");
		error = cudaMemcpy(device_fields[gpuIndex], host_fields[gpuIndex], num_bytes, cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {
                        printf("CUDA error: %s\n", cudaGetErrorString(error));
                }

		curandState *randomState;
		cudaMalloc((void**)&randomState, total_threads * sizeof(curandState));

		//hello<<<1,1>>>();
		tracer<<<num_blocks, num_threads>>>(device_fields[gpuIndex], device_vertices, device_faces, device_normals, device_colors, q, randomState, device_params);
	
	}

	error = cudaGetLastError();
        if (error != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(error));
        }

	printf("Kernels have started...\n");
	printf("Waiting for results...\n");
	int num_gpu_bytes = gpuFieldSize * sizeof(float) * 3;
	for (int gpuIndex = 0; gpuIndex < gpuCount; gpuIndex++) {
		cudaSetDevice(gpuIndex);
		
		 // Block and copy result
                error = cudaMemcpy(host_fields[gpuIndex], device_fields[gpuIndex], num_gpu_bytes, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess) {
                        printf("CUDA error: %s\n", cudaGetErrorString(error));
                }
		printf("Copied memory from: %d\n", device_fields[gpuIndex]);


                //Copy to the result to one result field
                for (int i = 0; i < gpuFieldSize; i++) {
                        //printf("copy to %d\n", i + gpuFieldSize * gpuIndex);
                        result_field[i + 3 * gpuFieldSize * gpuIndex] = host_fields[gpuIndex][i];
                }

	}

	std::clock_t timeStop = clock();
	const double timeDifference = double(timeStop - timeStart) / CLOCKS_PER_SEC;
	std::cout << "Kernel took " << timeDifference << "s.\n";



	char buff[100];
	std::ofstream outputFile;
	sprintf(buff, "output/2/ouput-%dx%d-i%d-b%d.csv", WIDTH, HEIGHT, ITERATIONS, BOUNCES);
	outputFile.open(buff, std::ios::trunc);

	for (int i = 0; i < HEIGHT; i++) {
                for (int j = 0; j < WIDTH * 3; j++) {
			if (size <= 1024) {
				printf("%.2f ", result_field[i * WIDTH * 3 + j]);
			}
			outputFile << result_field[i * WIDTH * 3 + j] << " ";
       	        }
		if (size <= 1024) {
			std::cout << std::endl;
		}
		outputFile << std::endl;
        }

	outputFile.close();

	error = cudaGetLastError();
        if (error != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(error));
        }


	return 0;
}

