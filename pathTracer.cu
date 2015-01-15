#include <stdio.h>
#include <iostream>
#include <math.h>

struct hit {
	bool isHit;
	int index;
	float t;
};

__device__ float innerProduct(float* u, float* v, int uIndex, int vIndex) {

	return u[uIndex] * v[vIndex] + u[uIndex+1] * v[vIndex+1] + u[uIndex+2] * v[vIndex+2];
}

__device__ void crossProduct(float* u, float* v, int uIndex, int vIndex, float* result) {
	result[0] = u[vIndex + 1] * v[uIndex + 2] - u[vIndex + 2] * v[uIndex + 1]; //v X v2
        result[1] = u[vIndex + 2] * v[uIndex + 0] - u[vIndex + 0] * v[uIndex + 2];
        result[2] = u[vIndex + 0] * v[uIndex + 1] - u[vIndex + 1] * v[uIndex + 0];
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
        for (int i = 0; i < (3 * 10); i += 3) {
		//printf("%d:\n", i);
		//printf("Face [%d, %d, %d]\n", faces[i], faces[i+1], faces[i+2]);
		//printf("Triangle: [%f, %f, %f][%f, %f, %f][%f, %f, %f]", vertices[0],  vertices[1],  vertices[2],  vertices[3],  vertices[4],  vertices[5],  vertices[6],  vertices[7],  vertices[8]);

		triangle[0] = vertices[3 * faces[i + 0] + 0];
		triangle[1] = vertices[3 * faces[i + 0] + 1];
		triangle[2] = vertices[3 * faces[i + 0] + 2];
		triangle[3] = vertices[3 * faces[i + 1] + 0];
		triangle[4] = vertices[3 * faces[i + 1] + 1];
		triangle[5] = vertices[3 * faces[i + 1] + 2];
		triangle[6] = vertices[3 * faces[i + 2] + 0];
		triangle[7] = vertices[3 * faces[i + 2] + 1];
		triangle[8] = vertices[3 * faces[i + 2] + 2];

		//printf("Vertex [%f, %f, %f]\n", triangle[0], triangle[1], triangle[2]);
		//printf("Vertex [%f, %f, %f]\n", triangle[3], triangle[4], triangle[5]);
		//printf("Vertex [%f, %f, %f]\n", triangle[6], triangle[7], triangle[8]);



                intersection = checkItersection(rayStart, rayDirection, triangle, 0);
		/*if (rayStart[0] == 0.0 and rayStart[1] == 0.0) {
			printf("Vertex [%f, %f, %f]\n", triangle[0], triangle[1], triangle[2]);
		}*/
                if (intersection < closestIntersection) {
                        closestIntersection = intersection;
                        closestIntersectionIndex = i;
			//printf("Vertex [%f, %f, %f]\n", triangle[0], triangle[1], triangle[2]);
                	//printf("Vertex [%f, %f, %f]\n", triangle[3], triangle[4], triangle[5]);
                	//printf("Vertex [%f, %f, %f]\n", triangle[6], triangle[7], triangle[8]);

                }
        }

	hit rayHit = hit();
	rayHit.isHit = (closestIntersectionIndex != -1);
	rayHit.t = closestIntersection;
	rayHit.index = closestIntersectionIndex;

	return rayHit;
}

__global__ void tracer(float* field, float* vertices, int* faces, float* normals) {
	//printf("Face: [%d, %d, %d]\n", faces[0], faces[1], faces[2]);
	//printf("Vertex: [%f, %f, %f]\n", vertices[0], vertices[1], vertices[2]);
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int xIndex = index % 32 - 16;
	int yIndex = 16 - (int)(index / 32);

	float lightDirection[3] = {0.707107, 0.0, 0.707107};

	//printf("AA");
	float rayStart[3] = {xIndex, yIndex, 10.0};
	float rayDirection[3] = {0.0, 0.0, -1.0};

	hit rayHit = castRay(rayStart, rayDirection, vertices, faces);
	if (rayHit.isHit) {
		//field[index] = 1.0;
		//printf("Normal: [%f, %f, %f]\n", normals[rayHit.index], normals[rayHit.index + 1], normals[rayHit.index + 2]);
		//printf("Ray [%d, %d]", xIndex, yIndex);
		//printf("dot product: %f \n", innerProduct(normals, lightDirection, rayHit.index, 0));
		field[index] = innerProduct(normals, lightDirection, rayHit.index, 0);
	} else {
		field[index] = 0.0;
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
	int width = 32;
	int height = 32;

	const int trianglesCount = 10;
	float vertices[8 * 3] = {
		-10.0, -10.0,   0.0, //0  //Left wall
		-10.0, -10.0, -10.0, //1 
		-10.0,  10.0, -10.0, //2
		-10.0,  10.0,   0.0, //3
		 10.0, -10.0,   0.0, //4  //Right wall
                 10.0, -10.0, -10.0, //5
                 10.0,  10.0, -10.0, //6
                 10.0,  10.0,   0.0  //7
	};
	int faces[trianglesCount * 3] = {
		0, 1, 2, //Left wall
		0, 2, 3,
		1, 5, 6, //Back wall
		1, 6, 2,
		5, 7, 6, //Right wall
		5, 4, 7,
		2, 7, 3, //Top wall
		2, 6, 7,
		0, 5, 1, //Bottom wall
		0, 4, 5
	};
	float normals[trianglesCount * 3];
	for (int i = 0; i < trianglesCount * 3; i += 3) {
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
	float *device_vertices = 0;
	int *device_faces = 0;
	float *device_normals = 0;

	cudaMalloc((void**)&device_array, num_bytes);

	float faces_num_bytes = trianglesCount * 3 * sizeof(int);
	float vertices_num_bytes = 3 * 8 * sizeof(float);
	float normals_num_bytes = trianglesCount * 3 * sizeof(float);
	cudaMalloc((void**)&device_vertices, vertices_num_bytes);
	cudaMalloc((void**)&device_faces, faces_num_bytes);
	cudaMalloc((void**)&device_normals, normals_num_bytes);

	printf("Int: %i\n", sizeof(int));

	cudaMemcpy(device_faces, faces, faces_num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_vertices, vertices, vertices_num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_normals, normals, normals_num_bytes, cudaMemcpyHostToDevice);

	//hello<<<1,1>>>();
	//tracer<<<1,1>>>(device_array, device_vertices, device_faces, device_normals);
	tracer<<<num_blocks, num_threads>>>(device_array, device_vertices, device_faces, device_normals);
	cudaDeviceSynchronize();

	// download and inspect the result on the host:
	cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);


	for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
			//std::cout << result[i][j] << ", ";
			//std::cout << host_array[i * width + j] << " ";
			printf("%.2f ", host_array[i * width + j]);
                }
		std::cout << std::endl;
        }

	return 0;
}

