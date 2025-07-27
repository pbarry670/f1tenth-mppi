#include <torch/extension.h>
#include <cuda.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <stdio.h>


#define CUDA_NUM_THREADS  100
#define CUDA_NUM_BLOCKS  20
#define MAXIMUM_NUM_BLOCKS  4096

inline int GET_BLOCKS(const int N) {
  return std::max(std::min((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
           MAXIMUM_NUM_BLOCKS), 1);
}
//Define kernel function

//template <typename scalar_t>
//__global__ void time_horizon_rollout_kernel(
//    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
//    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> y,
//    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> psi,
//    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> vx,
//    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> vy,
//    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> u_steer,
//    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> u_throttle,
//    float u_steer_prev,
//    float u_throttle_prev,
//    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> centerline_x,
//    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> centerline_y,
//   torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> cost,
//    int numThreads) 
//{

template <typename scalar_t>
__global__ void time_horizon_rollout_kernel(
     scalar_t* __restrict__ x,
     scalar_t* __restrict__ y,
     scalar_t* __restrict__ psi,
     scalar_t* __restrict__ vx,
     scalar_t* __restrict__ vy,
     scalar_t* __restrict__ steerang,
     scalar_t* __restrict__ u_steerdot,
     scalar_t* __restrict__ u_throttle,
     float u_steerdot_prev,
     float u_throttle_prev,
     float pos_cost_scaler,
     float heading_cost_scaler,
     float ctrl_input_cost_scaler,
     float vel_cost_scaler,
     float V_desired,
     scalar_t* __restrict__ centerline_x,
     scalar_t* __restrict__ centerline_y,
     scalar_t* __restrict__ cost,
     int numThreads)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int i = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    scalar_t beta = 0; //Initialize dynamics variables
    scalar_t x_dot = 0;
    scalar_t y_dot = 0;
    scalar_t psi_dot = 0;

    scalar_t step_cost = 0; //Initialize cost for one step in the time horizon

    scalar_t min_dist = 0.0; //Initialize variable to store minimum distance from car to a centerline point
    int index_of_min_dist = 0; //Index of centerline point with minimum distance to car

    scalar_t pos_cost = 0.0; //Initialize cost variables
    scalar_t heading_cost = 0.0;
    scalar_t vel_cost = 0.0;
    scalar_t ctrl_cost = 0.0;

    scalar_t DT = 0.06; //Time step for the time horizon
    scalar_t L_R = 0.1651; //rear axle to cg, m
    scalar_t L_F = 0.1651; //front axle to cg, m
    int HORIZON_TIME_STEPS = 20; //number of time steps in time horizon
    int NUM_TRAJECTORIES = 2000; //number of trajectories. Should be equal to CUDA_NUM_THREADS*CUDA_NUM_BLOCKS, defined at the top of this file
    scalar_t RACELINE_WIDTH = 1.0; //side-to-side length of track; acts as a normalization factor for position cost

    scalar_t POS_COST_SCALER = pos_cost_scaler; //Initialize cost parameters for each type of cost
    scalar_t HEADING_COST_SCALER = heading_cost_scaler;
    scalar_t V_DESIRED = V_desired;
    scalar_t VEL_COST_SCALER = vel_cost_scaler;
    scalar_t CTRL_INPUT_COST_SCALER = ctrl_input_cost_scaler;
    
    for (int j = 0; j < HORIZON_TIME_STEPS; j++)
    {
        scalar_t u_stdot = u_steerdot[i + j*NUM_TRAJECTORIES]; //Obtain the control inputs for the given trajectory at the given time step
        scalar_t u_th = u_throttle[i + j*NUM_TRAJECTORIES];
        
        
        //Dynamics
        beta = atanf((L_R / (L_F + L_R))*tanf(steerang[i]));
        x_dot = vx[i]*cosf(psi[i] + beta);
        y_dot = vx[i]*sinf(psi[i] + beta);
        psi_dot = (vx[i]*cosf(beta)*tanf(steerang[i])) / (L_R + L_F);

        x[i] = x[i] + x_dot*DT; //Update x position of car (longitudinal)
        y[i] = y[i] + y_dot*DT; //Update y position of car (lateral)
        psi[i] = psi[i] + psi_dot*DT; //Update heading angle of car
        vx[i] = vx[i] + u_th*DT; //Update longitudinal velocity of car
        steerang[i] = steerang[i] + u_stdot*DT; //Update the steering angle of the car


        //Position cost start
        for (int k = 0; k < sizeof(centerline_x); k++) //Find closest point on centerline
        {
            scalar_t dist = (x[i] - centerline_x[k])*(x[i] - centerline_x[k]) + (y[i] - centerline_y[k])*(y[i] - centerline_y[k]);
            if (k == 0)
            { //If centerline point considered is the first point on the centerline, call it the minimum distance to start
                min_dist = dist;
                index_of_min_dist = 0;
            }
            else if (dist < min_dist)
            { //If a centerline point if found with less distance to the car than any before it, update min_dist and its index
                min_dist = dist;
                index_of_min_dist = k;
            }
        }
        pos_cost = ((y[i] - centerline_y[index_of_min_dist])/RACELINE_WIDTH)*((y[i] - centerline_y[index_of_min_dist])/RACELINE_WIDTH); //Position cost is lateral distance from centerline, normalized by raceline width, squared 
        pos_cost = POS_COST_SCALER*pos_cost;
        //Position cost end

        //Heading cost start
        if (index_of_min_dist == sizeof(centerline_x) - 1) //If closest centerline point is the last centerline point...
        {
            scalar_t dy = centerline_y[sizeof(centerline_x) - 1] - centerline_y[sizeof(centerline_x) - 2];
            scalar_t dx = centerline_x[sizeof(centerline_x) - 1] - centerline_x[sizeof(centerline_x) - 2];
            scalar_t ideal_heading = atan2f(dy, dx); 
            heading_cost = (psi[i] - ideal_heading)*(psi[i] - ideal_heading);       
            //If no centerline available, assume ideal heading is heading of final centerline points
        }
        else
        {
            scalar_t dy = centerline_y[index_of_min_dist+1] - centerline_y[index_of_min_dist];
            scalar_t dx = centerline_x[index_of_min_dist+1] - centerline_x[index_of_min_dist];

            scalar_t ideal_heading = atan2f(dy, dx);
            heading_cost = (psi[i] - ideal_heading)*(psi[i] - ideal_heading);
            //Heading cost is difference in car heading with path between closest centerline point and next centerline point
        }
        heading_cost = HEADING_COST_SCALER*heading_cost;
        //Heading cost end

        //Velocity cost start
        vel_cost = (V_DESIRED - vx[i])*(V_DESIRED - vx[i]); //Velocity cost is the difference from desired velocity squared
        vel_cost = VEL_COST_SCALER*vel_cost;
        //Velocity cost end

        //Control input cost start
        ctrl_cost = (u_stdot - u_steerdot_prev)*(u_stdot - u_steerdot_prev) + (u_th - u_throttle_prev)*(u_th - u_throttle_prev);
        ctrl_cost = CTRL_INPUT_COST_SCALER*ctrl_cost; //Control input cost is deviation from previous control input squared
        //Control input cost end

        u_steerdot_prev = u_stdot; //Acknowledge passage of one time step
        u_throttle_prev = u_th;
        
        //Add all costs up for this particular time step
        step_cost = pos_cost + heading_cost + vel_cost + ctrl_cost;
        cost[i] = cost[i] + step_cost; //Store overall cost of this trajectory here
    }
    
    
    __syncthreads();
    //printf("Total Cost: %f, Trajectory: %d \n", cost[i], i);
    
}

std::vector<torch::Tensor> time_horizon_rollout_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor psi,
    torch::Tensor vx,
    torch::Tensor vy,
    torch::Tensor steerang,
    torch::Tensor u_steer,
    torch::Tensor u_throttle,
    float u_steer_prev,
    float u_accel_prev,
    float pos_cost_scaler,
    float heading_cost_scaler,
    float ctrl_input_cost_scaler,
    float vel_cost_scaler,
    float V_desired,
    torch::Tensor centerline_x,
    torch::Tensor centerline_y,
    torch::Tensor cost
) {
    int numThreads = sizeof(x);
      
     AT_DISPATCH_FLOATING_TYPES(at::ScalarType::Float, "time_horizon_rollout_cuda", ([&] {
       time_horizon_rollout_kernel<scalar_t><<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(
        x.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        psi.data_ptr<scalar_t>(),
        vx.data_ptr<scalar_t>(),
        vy.data_ptr<scalar_t>(),
        steerang.data_ptr<scalar_t>(),
        u_steer.data_ptr<scalar_t>(),
        u_throttle.data_ptr<scalar_t>(),
        u_steer_prev,
        u_accel_prev,
        pos_cost_scaler,
        heading_cost_scaler,
        ctrl_input_cost_scaler,
        vel_cost_scaler,
        V_desired,
        centerline_x.data_ptr<scalar_t>(),
        centerline_y.data_ptr<scalar_t>(),
        cost.data_ptr<scalar_t>(),
        numThreads);
     }));

    cudaDeviceSynchronize();
    return {cost};

}

