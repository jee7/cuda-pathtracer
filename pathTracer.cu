#include <stdio.h>
#include <iostream>
#include <math.h>

#include "cudaQueue.cu"
#include <curand.h>
#include <curand_kernel.h>

#define ITERATIONS 10
#define BOUNCES 3 //At least 3!!
#define WIDTH 32
#define HEIGHT 32

struct hit {
	bool isHit;
	int index;
	float t;
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
	//printf("BB");

	float intersection;
        float closestIntersection = INFINITY;
        int closestIntersectionIndex = -1;
	float triangle[9];
        for (int i = 0; i < (3 * 12); i += 3) {
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


__global__ void rayTrace(float* bufferField, float* field, float* vertices, int* faces, float* normals, Queue* q, curandState* state) {
	//printf("pop-");
	Task task = q->pop();
	//printf("popped\n");
	if (!task.isValid) {
		//printf("Invalid task!\n");
		return;
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
	if (depth == 0) {
		//bufferField[index] = 1.0;
	}
	if (depth >= BOUNCES - 1) { //Max depth, add our value
        //        bufferField[index] = 0.0;
		atomicAdd(&(field[index]), task.value / ITERATIONS);
		return;
        }

	hit rayHit = castRay(rayStart, rayDir, vertices, faces);
        if (rayHit.isHit) {
                if (rayHit.index >= 10 * 3) {
			if (depth == 0) {
				atomicExch(&(field[index]), 1.0);
			}
                        //bufferField[index] *= 1.0;
                } else {
                        //field[index] = innerProduct(normals, lightDirection, rayHit.index, 0);
			//Generate random ray
			//BRDF with it
			float randomRayDir[3];
			randomReflect(normals, rayHit.index, randomRayDir, state);
			float brdf = innerProduct(normals, randomRayDir, rayHit.index, 0);
			//bufferField[index] *= brdf;
			/*if (depth == 2) {
				printf("Ray [%f, %f, %f] - ", rayStart[0], rayStart[1], rayStart[2]);
				printf("brdf: %f", brdf);
				printf(" - value: %f", bufferField[index]);
				printf("\n");
			}*/

			rayStart[0] += rayHit.t * rayDir[0];
			rayStart[1] += rayHit.t * rayDir[1];
			rayStart[2] += rayHit.t * rayDir[2];
			
			/*Task newTask = Task();
	                newTask.data[0] = rayStart[0];
	                newTask.data[1] = rayStart[1];
	                newTask.data[2] = rayStart[2];
	                newTask.data[3] = randomRayDir[0];
	                newTask.data[4] = randomRayDir[1];
	                newTask.data[5] = randomRayDir[2];
			newTask.index   = index;
			newTask.depth   = depth + 1;
			newTask.value   = task.value * brdf;
        	        newTask.isValid = true;
			q->push(newTask);*/
			
			task.data[0] = rayStart[0];
			task.data[1] = rayStart[1];
			task.data[2] = rayStart[2];
			task.data[3] = randomRayDir[0];
			task.data[4] = randomRayDir[1];
			task.data[5] = randomRayDir[2];
			task.depth  += 1;
			task.value  *= brdf;

	                q->push(task);
                }
        } else {
                //bufferField[index] = 0.0;
        }
}


__global__ void tracer(float* field, float* bufferField, float* vertices, int* faces, float* normals, Queue* q, curandState* state) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int xIndex = index % HEIGHT - WIDTH / 2;
	int yIndex = HEIGHT / 2 - (int)(index / WIDTH);

	//float lightDirection[3] = {0.707107, 0.0, 0.707107};

	curand_init(1337 + index, 0, 0, &state[index]);

	float pixelCoordinates[3] = {xIndex, yIndex, 1.0};
	float rayStart[3] = {0.0, 0.0, 5.0};
	float rayDirection[3];
	rayDirection[0] = pixelCoordinates[0] - rayStart[0];
	rayDirection[1] = pixelCoordinates[1] - rayStart[1];
	rayDirection[2] = pixelCoordinates[2] - rayStart[2];

	for (int i = 0; i < ITERATIONS; i++) {
		//bufferField[index] = 1.0;

		Task task = Task();
		task.data[0] = rayStart[0];
		task.data[1] = rayStart[1];
		task.data[2] = rayStart[2];
		task.data[3] = rayDirection[0];
		task.data[4] = rayDirection[1];
		task.data[5] = rayDirection[2];
		task.index   = index;
		task.depth   = 0;
		task.value   = 1.0;
		task.isValid = true;

		q->push(task);
		for (int bounce = 0; bounce < BOUNCES; bounce++) {
			//Send #aliveRay kernels
			//if (field[index] != 0.0) { //Alive ray
				rayTrace<<<1,1>>>(bufferField, field, vertices, faces, normals, q, state);
			//}
			//cudaDeviceSynchronize();
			//__syncthreads();
		}
		//field[index] += bufferField[index] / ITERATIONS;
		//atomicAdd(&(field[index]), bufferField[index] / ITERATIONS);
		//__syncthreads();
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
	int width = WIDTH;
	int height = HEIGHT;
	int size = width * height;

	const int trianglesCount = 12;
	const int verticesCount = 12 * 3; //This is not triangles count * 3, some triangles share
	const int facesCount = trianglesCount * 3; //Each triangle has a face

	float vertices[verticesCount] = {
		-10.0, -10.0,   6.0, //0  //Left wall
		-10.0, -10.0, -10.0, //1 
		-10.0,  10.0, -10.0, //2
		-10.0,  10.0,   6.0, //3
		 10.0, -10.0,   6.0, //4  //Right wall
                 10.0, -10.0, -10.0, //5
                 10.0,  10.0, -10.0, //6
                 10.0,  10.0,   6.0, //7
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
		/*0, 7, 3, //Front wall?
		0, 4, 7,*/
		8, 9, 10, //Light source
		8, 10, 11
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

	/*float near = 0.0;
	float far = 100.0;
	float left = width / -2.0;
	float right = width / 2.0;
	float top = height / 2.0;
	float bottom = height / -2.0;
	*/
	int q_size = size * BOUNCES * ITERATIONS;
	q_size = size * BOUNCES * ITERATIONS;

	Queue* q = 0;
	Queue* host_q = new Queue();
	host_q->init(q_size);
	cudaMalloc((void **)&q, sizeof(Queue));
	printf("Queue size: %d\n", sizeof(Queue));
	printf("Queue lists: %d\n", q_size * sizeof(Task));
	cudaError_t error = cudaMemcpy(q, host_q, sizeof(Queue), cudaMemcpyHostToDevice);

	if(error != cudaSuccess) {
	    printf("CUDA error: %s\n", cudaGetErrorString(error));
	}

	//Sick... http://stackoverflow.com/questions/16024087/copy-an-object-to-device
	Task* d_data = 0;
	//Task* d_out = 0;
	cudaMalloc((void **)&d_data, q_size * sizeof(Task));
	//cudaMalloc((void **)&d_out, q_size * sizeof(Task));

	cudaMemcpy(d_data, host_q->data, q_size * sizeof(Task), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_out, host_q->out, q_size * sizeof(Task), cudaMemcpyHostToDevice);

	cudaMemcpy(&(q->data), &d_data, sizeof(Task *), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(q->out), &d_out, sizeof(Task *), cudaMemcpyHostToDevice);

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}


	int num_bytes = size* sizeof(float);
	//int num_threads = 16;
	//int num_blocks = size / 16; //size / 128;
	int num_threads, num_blocks;
	if (size >= 128) {
		num_threads = 128;
		num_blocks = size / 128;
	} else if (size >= 16) {
		num_threads = 16;
		num_blocks = size / 16;
	} else {
		num_threads = size;
		num_blocks = 1;
	}
	printf("Blocks: %d, ThreadPerBlock: %d\n", num_blocks, num_threads);
	//int num_threads = size;
	//int num_blocks = 1;

	/*
		Threads in different blocks cannot
		synchronize -> CUDA runtime system
		can execute blocks in any order
	*/

	float *host_field = 0;
	host_field = (float*)malloc(num_bytes);
	for (int i = 0; i < size; i++) {
		host_field[i] = 0.0; //We start with 0.0 
	}

	// cudaMalloc a device array
	float *device_field = 0;
	float *device_buffer_field = 0;
	float *device_vertices = 0;
	int *device_faces = 0;
	float *device_normals = 0;

	cudaMalloc((void**)&device_field, num_bytes);
	cudaMalloc((void**)&device_buffer_field, num_bytes);
	//cudaMemset(&device_buffer_field


	float faces_num_bytes = facesCount * sizeof(int);
	float vertices_num_bytes = verticesCount * sizeof(float);
	float normals_num_bytes = facesCount * sizeof(float);
	cudaMalloc((void**)&device_vertices, vertices_num_bytes);
	cudaMalloc((void**)&device_faces, faces_num_bytes);
	cudaMalloc((void**)&device_normals, normals_num_bytes);

	printf("Int: %i\n", sizeof(int));

	cudaMemcpy(device_faces, faces, faces_num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_vertices, vertices, vertices_num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_normals, normals, normals_num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_field, host_field, num_bytes, cudaMemcpyHostToDevice);
	//cudaMemcpy(device_buffer_field, host_buffer_field, num_bytes, cudaMemcpyHostToDevice);


	curandState *randomState;
	cudaMalloc((void**)&randomState, size * sizeof(curandState));

	cudaDeviceSynchronize();


	//hello<<<1,1>>>();
	//tracer<<<1,5>>>(device_array, device_vertices, device_faces, device_normals, q);
	tracer<<<num_blocks, num_threads>>>(device_field, device_buffer_field, device_vertices, device_faces, device_normals, q, randomState);
	cudaDeviceSynchronize();

	// download and inspect the result on the host:
	cudaMemcpy(host_field, device_field, num_bytes, cudaMemcpyDeviceToHost);


	for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
			//std::cout << result[i][j] << ", ";
			//std::cout << host_array[i * width + j] << " ";
			printf("%.1f ", host_field[i * width + j]);
                }
		std::cout << std::endl;
        }

	error = cudaGetLastError();
        if (error != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(error));
        }


	return 0;
}

