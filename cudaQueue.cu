struct Task {
	float data[6];
	float value[3];
	bool isValid;
	int index;
	short depth; 
};

class Queue {

	int i;
	int j;
	int lock;

	public:
		Task* data;
		int size;

		__host__ __device__ void init(int size);
		__device__ void push(Task t);
		__device__ Task pop();
	private:
		__device__ void swap();

};


__host__ __device__ void Queue::init(int size) {
	this->i = 0;
	this->j = 0;
	this->lock = 0;

	this->data = new Task[size];
	this->size = size;
	
	printf("Init queue, i=%d", i);
}

__device__ void Queue::push(Task t) {

	int currentJ = atomicAdd(&j, 1);

	if (currentJ - i >= size) {
		printf("Queue is full!!\n");
	} else {
		int index = (currentJ) % size;
		data[index] = t;
		//printf("Push (%d)\n", index);
		//printf("Added to q (%d), i=%d\n", t.index, currentI);
		//printf("%d, ", currentI);
	}
}

__device__ Task Queue::pop() {

	Task resultTask;

	int currentI = atomicAdd(&i, 1);
	if (currentI >= j) {
		//printf("Queue is empty (i=%d j=%d)", i, j);
		resultTask = Task();
		resultTask.isValid = false;
	} else {
		int index = currentI % size;
		//printf("Pop (%d)\n", index);

		Task a = Task(); a.isValid = false;
		resultTask = data[index];
		data[index] = a;
		
	}

	return resultTask;
}
