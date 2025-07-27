from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='time_horizon_cuda',
    ext_modules=[
        CUDAExtension('time_horizon_cuda', ['time_horizon_rollout_cuda.cpp', 'time_horizon_rollout_kernel.cu',]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
