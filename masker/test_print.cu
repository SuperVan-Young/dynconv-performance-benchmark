
#include <stdio.h>

__global__ void  print_kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}
__global__ void block_sym_kernel(float* a, float* out, float* mask) {
    // printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
    if (mask[blockIdx.x]>0){
    // if (1){
        out[blockIdx.x*(300)+threadIdx.x]=0;
        for(int i=0;i<400;i++){
            out[blockIdx.x*(300)+threadIdx.x]+=a[blockIdx.x*(300*400)+threadIdx.x*(400)+i];
        }
    }
}

int main() {
    dim3 grid(200);
    dim3 block(300);
    float *a,*mask,*out;
    cudaMalloc(&a, 200*300*400*sizeof(float));
    cudaMalloc(&mask, 200*sizeof(float));
    cudaMalloc(&out, 200*300*sizeof(float));


    for(int n_pos=0;n_pos<10;n_pos++){
        float *mask_cpu;
        mask_cpu=new float [200];
        int pos=0;
        for (int i=0;i<200;i++){
            mask_cpu[i]=float(rand()%10)-n_pos;
            // printf("%f ",mask_cpu[i]);
            if(mask_cpu[i]>0){
                pos+=1;
            }
        }
        // printf("\npos_frac=%f\n",float(pos)/200);
        // for (int i=0;i<200;i+=2){
        //     mask_cpu[i]=1.f;
        // }
        // for (int i=1;i<200;i+=2){
        //     mask_cpu[i]=-1.f;
        // }
        cudaMemcpy(mask,mask_cpu, 200*sizeof(float), cudaMemcpyHostToDevice);

        float tot_time=0.f;
        int repeat=10;
        for(int i=0;i<repeat;i++){
            cudaEvent_t start1;  
            cudaEventCreate(&start1);  
            cudaEvent_t stop1;  
            cudaEventCreate(&stop1);  
            cudaEventRecord(start1, NULL);  
            block_sym_kernel<<<grid,block>>>(a,out,mask);
            cudaDeviceSynchronize();
            cudaEventRecord(stop1, NULL);  
            cudaEventSynchronize(stop1);
            float msecTotal1 = 0.0f;  
            cudaEventElapsedTime(&msecTotal1, start1, stop1);
            // printf("%d %f\n",i,msecTotal1);
            tot_time+=msecTotal1;
        }
        // printf("pos_frac=%f\n",float(pos)/200);
        // printf("tot %f\n",tot_time/repeat);
        printf("%f\t%f\n",float(pos)/200,tot_time/repeat);
    }
    // print_kernel<<<grid, block>>>();
    // cudaDeviceSynchronize();
}