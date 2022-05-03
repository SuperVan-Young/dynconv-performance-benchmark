#include "time.h"
#include <stdio.h>


#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
// extern "C" 
// __global__ void __launch_bounds__(150) default_function_kernel0(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output) {
extern "C" __global__ void __launch_bounds__(160) default_function_kernel0(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output) {
// extern "C" __global__ void default_function_kernel0(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output) {
  float output_local[4];
  __shared__ float input_shared[256];
  __shared__ float weight_shared[160];
  float input_shared_local[2];
  float weight_shared_local[8];
  output_local[0] = 0.000000e+00f;
  output_local[1] = 0.000000e+00f;
  output_local[2] = 0.000000e+00f;
  output_local[3] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((((int)threadIdx.z) * 52) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 256) {
        if (((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 13) + ((int)threadIdx.y)) < 4) {
          if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 13) {
            input_shared[((((((int)threadIdx.z) * 52) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = input[((((((rc_outer * 25088) + ((((((((int)threadIdx.z) * 52) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 5) * 3136)) + (((int)blockIdx.y) * 224)) + (((((((((int)threadIdx.z) * 52) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 31) >> 3) * 56)) + (((int)blockIdx.x) * 8)) + (((((((int)threadIdx.z) * 52) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 7))];
          }
        }
      }
    }
    if (((((int)blockIdx.z) * 4) + (((((int)threadIdx.z) * 4) + ((int)threadIdx.y)) / 5)) < 13) {
      weight_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = weight[(((((((int)blockIdx.z) * 1280) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (rc_outer * 8)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int rc_inner_outer = 0; rc_inner_outer < 4; ++rc_inner_outer) {
      for (int ax1 = 0; ax1 < 2; ++ax1) {
        input_shared_local[ax1] = input_shared[((((rc_inner_outer * 64) + (ax1 * 32)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))];
      }
      for (int ax11 = 0; ax11 < 2; ++ax11) {
        weight_shared_local[ax11] = weight_shared[(((((int)threadIdx.z) * 8) + (rc_inner_outer * 2)) + ax11)];
        if (((int)blockIdx.z) < 3) {
          weight_shared_local[(ax11 + 2)] = weight_shared[((((((int)threadIdx.z) * 8) + (rc_inner_outer * 2)) + ax11) + 40)];
        }
        if (((int)blockIdx.z) < 3) {
          weight_shared_local[(ax11 + 4)] = weight_shared[((((((int)threadIdx.z) * 8) + (rc_inner_outer * 2)) + ax11) + 80)];
        }
        if (((int)blockIdx.z) < 3) {
          weight_shared_local[(ax11 + 6)] = weight_shared[((((((int)threadIdx.z) * 8) + (rc_inner_outer * 2)) + ax11) + 120)];
        }
      }
      for (int rc_inner_inner = 0; rc_inner_inner < 2; ++rc_inner_inner) {
        output_local[0] = (output_local[0] + (input_shared_local[rc_inner_inner] * weight_shared_local[rc_inner_inner]));
        if (((int)blockIdx.z) < 3) {
          output_local[1] = (output_local[1] + (input_shared_local[rc_inner_inner] * weight_shared_local[(rc_inner_inner + 2)]));
        }
        if (((int)blockIdx.z) < 3) {
          output_local[2] = (output_local[2] + (input_shared_local[rc_inner_inner] * weight_shared_local[(rc_inner_inner + 4)]));
        }
        if (((int)blockIdx.z) < 3) {
          output_local[3] = (output_local[3] + (input_shared_local[rc_inner_inner] * weight_shared_local[(rc_inner_inner + 6)]));
        }
      }
    }
  }
  output[((((((((int)blockIdx.z) * 62720) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))] = output_local[0];
  if (((int)blockIdx.z) < 3) {
    output[(((((((((int)blockIdx.z) * 62720) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 15680)] = output_local[1];
  }
  if (((int)blockIdx.z) < 3) {
    output[(((((((((int)blockIdx.z) * 62720) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 31360)] = output_local[2];
  }
  if (((int)blockIdx.z) < 3) {
    output[(((((((((int)blockIdx.z) * 62720) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 47040)] = output_local[3];
  }
}



int pre_mask_to_ind(float* pre_mask, int* indx, int* indy, int H,int W, int granularity){
    float* tmp;
    printf("premask[0] %f ",pre_mask[0]);

    // printf("W=%d granularity=%d",W,granularity);
    tmp=new float [W/granularity];
    for(int i =0;i<W/granularity;i++){
        // printf("i=%d ",i);
        tmp[i]=0.f;
    }
    int cnt=0;
    for(int y =0;y<H;y++){
        // printf("premask[y,0] %f ",pre_mask[y*W]);
        for(int x=0;x<W;x++){
            tmp[x/granularity]+=pre_mask[y*W+x];
        }
        if(y%granularity==(granularity-1)){
            for(int i=0;i<W/granularity;i++){
                
                if (tmp[i]>0){
                    indx[cnt]=i;
                    indy[cnt]=y/granularity;
                    cnt++;
                }
                tmp[i]=0;
            }
        }
    }
    return cnt;
}


int main(){
    int H=56;
    int W=56;
    int channel=64;
    int granularity=7;

    float *output,*output_cuda;
    int N=203840;
    output=new float [N];
    cudaMalloc(&output_cuda, N*sizeof(float));
    
    

    float *input,*input_cuda;
    N=H*H*channel;
    input=new float [N];
    for (int i=0;i<N;i++){
        input[i]=1.f;
    }
    cudaMalloc(&input_cuda, N*sizeof(float));
    cudaMemcpy(input_cuda,input, N*sizeof(float), cudaMemcpyHostToDevice);
    

    float *weight,*weight_cuda;
    N=channel*(channel+1);
    weight=new float [N];
    for (int i=0;i<N;i++){
        weight[i]=1.f;
    }
    cudaMalloc(&weight_cuda, N*sizeof(float));
    cudaMemcpy(weight_cuda,weight, N*sizeof(float), cudaMemcpyHostToDevice);
    
    
    float *pre_mask;
    
    int N_pre_mask=H*W;
    pre_mask=new float [N_pre_mask];
    for (int i=0;i<N_pre_mask;i++){
        pre_mask[i]=1.f;
    }
    int N_ind=H/granularity;
    int  *ind;
    ind=new int [2*N_ind];
    
    dim3 grid(7,14,4); 
    dim3 block(8,4,5);
    int repeat=100;
    float tot_time=0.f;
    // warm up
    for (int i=0;i<10;i++){
        default_function_kernel0<<<grid,block>>>(input_cuda,weight_cuda,output_cuda);
    }
    for(int i=0;i<repeat;i++){
        cudaEvent_t start1;  
        cudaEventCreate(&start1);  
        cudaEvent_t stop1;  
        cudaEventCreate(&stop1);  
        cudaEventRecord(start1, NULL);  
        default_function_kernel0<<<grid,block>>>(input_cuda,weight_cuda,output_cuda);
        // cudaDeviceSynchronize();
        cudaMemcpy(pre_mask,output_cuda, N_pre_mask*sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy(pre_mask,output_cuda, 2*sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop1, NULL);  
        cudaEventSynchronize(stop1);
        float msecTotal1 = 0.0f;  
        cudaEventElapsedTime(&msecTotal1, start1, stop1);
        printf("%d %f\n",i,msecTotal1);
        tot_time+=msecTotal1;
    }
    // int cnt=0;
    clock_t st=clock();
    // cudaMemcpy(pre_mask,output_cuda, N_pre_mask*sizeof(float), cudaMemcpyDeviceToHost);
    int cnt=pre_mask_to_ind(pre_mask,ind,ind+N_ind,H,H,granularity);
    clock_t ed=clock();
    double duration = double(ed - st) / CLOCKS_PER_SEC;
    printf("tot %f\n cnt=%d\n",tot_time/repeat,cnt);
    printf("duration %lf\n",duration*1000);
    
    return 0;
}
