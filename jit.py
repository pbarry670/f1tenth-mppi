from torch.utils.cpp_extension import load

time_horizon_cuda = load('time_horizon_cuda', ['time_horizon_rollout_cuda.cpp', 'time_horizon_rollout_kernel.cu'], verbose=True)
help(time_horizon_cuda)
