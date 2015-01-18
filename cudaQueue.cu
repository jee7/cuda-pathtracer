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
		__device__ bool push(Task t);
		__device__ Task pop();
		__device__ int length();
	private:
		__device__ void swap();

};


__host__ __device__ void Queue::init(int size) {
	this->i = 0;
	this->j = 0;
	this->lock = 0;

	this->data = new Task[size];
	this->size = size;
	
	printf("Init queue, i=%d, size=%llu\n", i, size);
}

__device__ bool Queue::push(Task t) {

	int currentJ = atomicAdd(&j, 1);

	if (currentJ - i >= size) {
		//printf("Queue is full at %d (i = %d)!!\n", currentJ, i);
		atomicSub(&j, 1);

		return false;
	} else {
		int index = (currentJ) % size;
		if (index > size) {
			printf("%llu > %llu\n", index, size);
		}
		data[index] = t;
		//printf("Push (%d) - i = %d, j = %d\n", index, i, j);
		//printf("Added to q (%d), i=%d\n", t.index, currentI);
		//printf("%d, ", currentI);

		return true;
	}
}

__device__ Task Queue::pop() {

	Task resultTask;

	int currentI = atomicAdd(&i, 1);
	if (currentI >= j) {
		atomicSub(&i, 1);
		//printf("Queue is empty (i=%d j=%d)", i, j);
		resultTask = Task();
		resultTask.isValid = false;
	} else {
		int index = currentI % size;
		//printf("Pop (%d) - i = %d, j = %d\n", index, i, j);

		Task a = Task(); a.isValid = false;
		resultTask = data[index];
		data[index] = a;
	}

	return resultTask;
}

__device__ int Queue::length() {

	return j - i;
}
