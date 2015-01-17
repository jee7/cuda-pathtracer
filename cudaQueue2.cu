struct Task {
	float data[6];
	bool isValid;
	float value;
	int index;
	short depth; 
};

class Queue {

//	Task* in;
//	Task* out;
	int i;
	int j;
	int lock;

	public:
		Task* in;
	        Task* out;
		int size;

		__host__ __device__ void init(int size);
		__device__ void push(Task t);
		__device__ Task pop();
	private:
		__device__ void swap();

};


__host__ __device__ void Queue::init(int size) {
	this->i = size;
	this->j = 0;
	this->lock = 0;

	this->in = new Task[size];
	this->out = new Task[size];
	
	this->size = size;
	
	//in = inValues;
	//out = outValues;
	printf("Init queue, i=%d", i);
}

__device__ void Queue::push(Task t) {
	while (atomicCAS(&lock, 0, 0) == 1) {};

	int currentI = atomicAdd(&i, -1);

	if (currentI == 0) {
		printf("Queue is full!!\n");
	} else {
		in[currentI - 1] = t;
		//printf("Added to q (%d), i=%d\n", t.index, currentI);
		//printf("%d, ", currentI);
	}
}

__device__ Task Queue::pop() {
	while (atomicCAS(&lock, 0, 0) == 1) {};

	if (j <= 0 && i < size) {
		swap();
	}
	Task resultTask;

	int currentJ = atomicAdd(&j, -1);
	if (currentJ <= 0) {
		//printf("Queue is empty (i=%d j=%d)", i, j);
		resultTask = Task();
		resultTask.isValid = false;
	} else {
		//printf("Pop, j=%d\n", currentJ);
		//printf("Pop (%d), j=%d\n", out[currentJ].index, currentJ);

		Task a = Task(); a.isValid = false;
		resultTask = out[currentJ];
		out[currentJ] = a;
		
	}

	return resultTask;
}

__device__ void Queue::swap() {
	//printf("Current lock: %d", atomicCAS(&lock, 0, 1));
	//atomicAdd(lock, 1);
	while(atomicCAS(&lock, 0, 1) == 1) {};
	printf("SWAP i=%d, j=%d, size=%d\n", i, j, size);

	if (j == 0 && i < size) {
		//printf("Swap:" );
	        //printf("j=%d, ", j);
	        //printf("i=%d\n", i);

		Task el;
		for(int u = 0; u < size; u++) {
			el = in[u];
			in[u] = out[size - u - 1];

			if (out[size - u - 1].isValid) {
				printf("Valid Task deleted...!");
			}
			out[size - u - 1] = el;
		}
		j = size - i;
		i = size;

		//printf("Swap:" );
	        //printf("j=%d, ", j);
	        //printf("i=%d\n", i);
	}
	atomicExch(&lock, 0);
	//atomicAdd(lock, -1);

}
