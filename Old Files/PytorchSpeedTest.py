import numpy as np
import torch
import time

numTimeSteps = 30
numTrajectories = 1000
dt = 0.03
totalTimeCPU = 0
totalTimeCUDA = 0
numTrials = 50

for i in range(numTrials - 1):

        x0_vec = torch.tensor(np.random.random((7, numTrajectories)))
        cost_vec = torch.tensor(np.zeros(numTrajectories))
        x0_vec_cuda = x0_vec.cuda()
        cost_vec_cuda = cost_vec.cuda()


        start_time = time.time()

        #GPU
        with torch.cuda.device(0):
                for i in range(numTimeSteps - 1): #Iterate numTimeSteps times
                        beta = torch.atanh((x0_vec_cuda[4,:]/((x0_vec_cuda[4,:] + x0_vec_cuda[5,:])))+ torch.tan(x0_vec_cuda[6,:]))

                        x_dot = x0_vec_cuda[2,:] * torch.cos(x0_vec_cuda[3,:]+beta)
                        y_dot = x0_vec_cuda[2,:] * torch.sin(x0_vec_cuda[3,:]+beta)
                        headAng_dot = (x0_vec_cuda[2,:]/(x0_vec_cuda[4,:] + x0_vec_cuda[5,:]))*torch.cos(beta)*torch.tan(x0_vec_cuda[6,:])

                        x0_vec_cuda[0,:] = x0_vec_cuda[0,:] + x_dot*dt
                        x0_vec_cuda[1,:] = x0_vec_cuda[1,:] + y_dot*dt
                        x0_vec_cuda[3,:] = x0_vec_cuda[3,:] + headAng_dot*dt
                        x0_vec_cuda[2,:] = torch.sqrt(x0_vec_cuda[0,:]*x0_vec_cuda[0,:] + x0_vec_cuda[1,:]*x0_vec_cuda[1,:])

                        #torch.cuda.synchronize()

                        cost_vec_cuda = cost_vec_cuda + (x0_vec_cuda[0,:])**2 + (x0_vec_cuda[1,:])**2
                        #torch.cuda.synchronize()

        end_time = time.time()
        dt_cuda = end_time - start_time
        print(f'cuda: {dt_cuda}')


        start_time = time.time()

        for i in range(numTimeSteps - 1): #Iterate numTimeSteps times
                beta = torch.atanh((x0_vec[4,:]/((x0_vec[4,:] + x0_vec[5,:])))+ torch.tan(x0_vec[6,:]))

                x_dot = x0_vec[2,:] * torch.cos(x0_vec[3,:]+beta)
                y_dot = x0_vec[2,:] * torch.sin(x0_vec[3,:]+beta)
                headAng_dot = (x0_vec[2,:]/(x0_vec[4,:] + x0_vec[5,:]))*torch.cos(beta)*torch.tan(x0_vec[6,:])

                x0_vec[0,:] = x0_vec[0,:] + x_dot*dt
                x0_vec[1,:] = x0_vec[1,:] + y_dot*dt
                x0_vec[3,:] = x0_vec[3,:] + headAng_dot*dt

                cost_vec = cost_vec + (x0_vec[0,:])**2 + (x0_vec[1,:])**2

        end_time = time.time()
        dt_cpu = end_time - start_time
        print(f'cpu: {dt_cpu}')

        totalTimeCPU += dt_cpu
        totalTimeCUDA += dt_cuda

cudaAvgTime = totalTimeCUDA/numTrials
CPUAvgTime = totalTimeCPU/numTrials
print('\n')
print(f'Trajectories: {numTrajectories}')
print(f'Time Steps: {numTimeSteps}')
print(f'CUDA Avg Time: {cudaAvgTime}')
print(f'CPU Avg Time: {CPUAvgTime}')
