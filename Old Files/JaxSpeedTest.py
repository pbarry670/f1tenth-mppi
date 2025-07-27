import jax 
import numpy as np
import time

numTimeSteps = 30
numTrajectories = 1000
dt = 0.03

state = state = np.random.random((numTrajectories, 7))
cost = np.zeros(numTrajectories)


def forward_dynamics(state, cost, dt):
    beta = jax.numpy.arctan((state[4,:]/(state[4,:] + state[5,:]))*jax.numpy.tan(state[6,:]))

    x_dot = state[2,:]*jax.numpy.cos(state[3,:] + beta)
    y_dot = state[2,:]*jax.numpy.sin(state[3,:] + beta)
    headAng_dot = (state[2,:]/(state[4,:]+state[5,:]))*jax.numpy.cos(beta)*jax.numpy.tan(state[6,:])

    state[0,:] = state[0,:] + x_dot*dt
    state[1,:] = state[1,:] + y_dot*dt
    state[3,:] = state[3,:] + headAng_dot*dt
    cost = cost + state[0,:]**2 + state[1,:]**2

    return state, cost

start_time = time.time()
forward_dynamics_jit = jit(forward_dynamics)

for i in range(numTimeSteps - 1):
    updatedState, updatedCost = forward_dynamics_jit(state, cost).block_until_ready()

    state = updatedState
    cost = updatedCost

end_time = time.time()

dt_cuda = end_time - start_time
print(f'cuda: {dt_cuda}')

start_time = time.time()
for i in range(numTimeSteps - 1):
    updatedState, updatedCost = forward_dynamics(state, cost)

    state = updatedState
    cost = updatedCost

end_time = time.time()
dt_cpu = end_time - start_time
print(f'cpu: {dt_cpu}')
print(f'Trajectories: {numTrajectories}')
print(f'Time Steps: {numTimeSteps}')
