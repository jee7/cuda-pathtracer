struct Task {
	bool isValid;
	float data[8];
};
#define MAX_SIZE 1024

class Queue {

	Task in[MAX_SIZE];
	Task out[MAX_SIZE];
	int i;
	int j;
	int lock;

	public:
		__host__ void init();
		__device__ void push(Task t);
		__device__ Task pop();
	private:
		__device__ void swap();

};


__host__ void Queue::init() {
	i = MAX_SIZE;
	j = 0;
	lock = 0;
	//in = inValues;
	//out = outValues;
	printf("Init queue, i=%d", i);
}

__device__ void Queue::push(Task t) {
	//while (atomicCAS(&lock, 0, 1) == 1) {};

	//in[i] = t;
        //i--;
	//atomicAdd(&i, -1);

	if (i == 0) {
		printf("Queue is full!!\n");
	} else {
		int currentI = atomicAdd(&i, -1) - 1;
		in[currentI] = t;
		//printf("Added to q (%f), i=%d\n", t.data[6], currentI);
		//printf("%d, ", currentI);
	}
	lock = 0;
}

__device__ Task Queue::pop() {
	//while (atomicCAS(&lock, 0, 1) == 1) {};
	while(lock == 1){}

	if (j == 0) {
		swap();
	}
	Task resultTask = Task();

	if (j <= 0) {
		//printf("Queue is empty");
		resultTask.isValid = false;
	} else {
		int currentJ = atomicAdd(&j, -1) - 1;
		//printf("Pop (%f), j=%d", out[currentJ].data[6], currentJ);
		resultTask = out[currentJ];
	}
	lock = 0;

	return resultTask;
}

__device__ void Queue::swap() {
	//printf("Current lock: %d", atomicCAS(&lock, 0, 1));
	//atomicAdd(lock, 1);
	//while (atomicCAS(&lock, 0, 1) == 1) {};

	if (j == 0 && i < MAX_SIZE) {
		/*printf("Swap:" );
	        printf("j=%d, ", j);
	        printf("i=%d\n", i);*/

		Task el;
		for(int u = 0; u < MAX_SIZE; u++) {
			el = in[u];
			in[u] = out[MAX_SIZE - u - 1];
			out[MAX_SIZE - u - 1] = el;
		}
		j = MAX_SIZE - i;
		i = MAX_SIZE;

		/*printf("Swap:" );
	        printf("j=%d, ", j);
	        printf("i=%d\n", i);*/
	}
	lock = 0;
	//atomicAdd(lock, -1);

}
