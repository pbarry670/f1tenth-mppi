#include <torch/extension.h>
#include <vector>

//CUDA forward declarations

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
  torch::Tensor cost);



//C++ interface
  //AT_ASSERT --> AT_CHECK ? Maybe.
  #define CHECK_CUDA(x) AT_ASSERTM(x.options().device().is_cuda(), #x "must be a CUDA tensor")
  #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")
  #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> time_horizon_rollout(
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
  torch::Tensor cost) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(psi);
    CHECK_INPUT(vx);
    CHECK_INPUT(vy);
    CHECK_INPUT(steerang);
    CHECK_INPUT(u_steer);
    CHECK_INPUT(u_throttle);
    CHECK_INPUT(centerline_x);
    CHECK_INPUT(centerline_y);
    CHECK_INPUT(cost);



    return time_horizon_rollout_cuda(x,y,psi,vx,vy,steerang,u_steer,u_throttle,u_steer_prev,u_accel_prev,pos_cost_scaler,heading_cost_scaler,ctrl_input_cost_scaler,vel_cost_scaler,V_desired,centerline_x,centerline_y,cost);
  }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("time_horizon_rollout", &time_horizon_rollout, "Time horizon rollout (CUDA)");
}
