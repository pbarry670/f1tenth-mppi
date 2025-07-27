#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

__global__ void compute_cost(float *cost, int numTrajectories, float randomNum1, float randomNum2){
    int i = threadIdx.x;

    if (i < numTrajectories)
        cost[i] = cost[i] + randomNum1*randomNum1 + randomNum2*randomNum2;

    __syncthreads();

}


__global__ void forward_dynamics(float *state, int dimStates, int numTrajectories){

    int i = threadIdx.x;

    float x = state[i*dimStates + 0]; 
    float y = state[i*dimStates + 1];
    float v = state[i*dimStates + 2];
    float headAng = state[i*dimStates + 3];
    float l_r = state[i*dimStates + 4];
    float l_f = state[i*dimStates + 5];
    float steerAng = state[i*dimStates + 6];


    float beta = atanf((l_r/(l_f+l_r))*tanf(steerAng));

    float x_dot = v*cosf(headAng+beta);
    float y_dot = v*sinf(headAng+beta);
    float headAng_dot = (v/(l_f + l_r))*cosf(beta)*tanf(steerAng);

    float x_new = x + x_dot*0.03;
    float y_new = y + y_dot*0.03;
    float headAng_new = headAng + headAng_dot*0.03;
    float v_new = sqrtf(x_new*x_new + y_new*y_new);

    state[i*dimStates + 0] = x_new;
    state[i*dimStates + 1] = y_new;
    state[i*dimStates + 2] = v_new;
    state[i*dimStates + 3] = headAng_new;

    __syncthreads();
}

int main(void){

    int numTrajectories = 8000;
    int dimStates = 7;
    int numTimeSteps = 100;

    float *h_state = new float[dimStates*numTrajectories];
    float *h_cost = new float[numTrajectories];

    float *d_state;
    float *d_cost;

    clock_t start, end;
    double time_taken;

    //Initialize both arrays
    for (int i = 0; i < numTrajectories; i++){
        h_cost[i] = 0;
        h_state[i] = (float)rand()/(float)(RAND_MAX); //Generate random number between 0 and 1
        
    }

    
    start = clock();

    for (int i = 0; i < numTimeSteps; i++){

        cudaMalloc((void**)&d_state, numTrajectories*dimStates*sizeof(float)); //Allocate memory in CUDA for state array
        cudaMemcpy(d_state, h_state, numTrajectories*dimStates*sizeof(float), cudaMemcpyHostToDevice); //Copy state array from host to device
        dim3 grid_size(1, 1, 1); dim3 block_size(numTrajectories, 1, 1); //Kernel parameters

        forward_dynamics <<<grid_size, block_size>>>(d_state, dimStates, numTrajectories); //Launch kernel for forwarding dynamics

        cudaMemcpy(h_state, d_state, numTrajectories*dimStates*sizeof(float), cudaMemcpyDeviceToHost); //Copy data back to host
        //cudaFree(d_state); //De-allocate memory
        //free(h_state);

        cudaMalloc((void**)&d_cost, numTrajectories*sizeof(float)); // Allocate memory in CUDA for cost array
        cudaMemcpy(d_cost, h_cost, numTrajectories*sizeof(float), cudaMemcpyHostToDevice); //Copy cost array from host to device

	float randomNum1 = (float)rand()/(float)(RAND_MAX);
	float randomNum2 = (float)rand()/(float)(RAND_MAX);
        compute_cost <<<grid_size, block_size>>>(d_cost, numTrajectories, randomNum1, randomNum2); //Launch kernel for computing cost

        cudaMemcpy(h_cost, d_cost, numTrajectories*sizeof(float), cudaMemcpyHostToDevice);
	//cudaFree(d_cost);
    	//free(h_cost);
        
        
    }
    cudaFree(d_state);
    free(h_state);

    cudaFree(d_cost);
    free(h_cost);

    end = clock();
    
    time_taken = ((double)(end - start))/CLOCKS_PER_SEC;

    printf("Time: %f seconds\n", time_taken);
 
    return 0;
}





