
#include <vector>

#include "caffe/layers/kernelrbf_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "stdio.h"


#include "iostream"
#include "fstream"
#include "stdlib.h"

namespace caffe {

using namespace std;

template<typename Dtype>
void KernelRbfLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    // write by gaozhi-------------
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* bottom_data_0 = bottom[0]->gpu_data();
    const Dtype* bottom_data_1 = bottom[1]->gpu_data();
    const int step_top = top[0]->count(1);
    const int step_bottom_0 = bottom[0]->count(1);
    const int step_bottom_1 = bottom[1]->count(1);
    const int hw_bottom_0 = bottom[0]->count(2);
    const int hw_bottom_1 = bottom[1]->count(2);

    const int channel_bottom_0=bottom[0]->shape(1);
    const int channel_bottom_1=bottom[1]->shape(1);

    Dtype p_beta=beta_;

    /*
    Dtype *m_fi2;
    Dtype *m_fj2;
    Dtype *m_fifj;
    Dtype *m_chw;
    Dtype *v_c_hw;
    */
    
    /*
    size_t pitch;
    CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&m_fi2),
                    &pitch,sizeof(Dtype)*channel_bottom_0,channel_bottom_0));
    CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&m_fj2),
                    &pitch,sizeof(Dtype)*channel_bottom_0,channel_bottom_0));
    CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&m_fifj),
                    &pitch,sizeof(Dtype)*channel_bottom_0,channel_bottom_0));

    CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&m_chw),
                    &pitch,sizeof(Dtype)*hw_bottom_0,channel_bottom_0));

    CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&v_c_hw),
                    &pitch,sizeof(Dtype)*hw_bottom_0,channel_bottom_0));

    */

    /*
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_fi2),
                    sizeof(Dtype)*channel_bottom_0*channel_bottom_0));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_fj2),
                    sizeof(Dtype)*channel_bottom_0*channel_bottom_0));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_fifj),
                    sizeof(Dtype)*channel_bottom_0*channel_bottom_0));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_chw),
                    sizeof(Dtype)*hw_bottom_0*channel_bottom_0));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&v_c_hw),
                    sizeof(Dtype)*hw_bottom_0*channel_bottom_0));
    */
    
    Blob<Dtype> b_m_fi2;
    Blob<Dtype> b_m_fj2;
    Blob<Dtype> b_m_fifj;
    Blob<Dtype> b_m_chw;
    Blob<Dtype> b_v_c_hw;

    b_m_fi2.Reshape(1,1,channel_bottom_0,channel_bottom_0);
    b_m_fj2.Reshape(1,1,channel_bottom_0,channel_bottom_0);
    b_m_fifj.Reshape(1,1,channel_bottom_0,channel_bottom_0);
    b_m_chw.Reshape(1,1,channel_bottom_0,hw_bottom_0);
    b_v_c_hw.Reshape(1,1,channel_bottom_0,hw_bottom_0);

    Dtype * m_fi2=b_m_fi2.mutable_gpu_data();
    Dtype * m_fj2=b_m_fj2.mutable_gpu_data();
    Dtype * m_fifj=b_m_fifj.mutable_gpu_data();
    Dtype * m_chw=b_m_chw.mutable_gpu_data();
    Dtype * v_c_hw=b_v_c_hw.mutable_gpu_data();

    for (int b = 0; b < bottom[0]->shape(0); ++b) {
        //fifi------------------
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channel_bottom_0,
                channel_bottom_1, hw_bottom_0, Dtype(1.0),
                bottom_data_0 + b * step_bottom_0,
                bottom_data_1 + b * step_bottom_1, Dtype(0.0),
                m_fifj);

        //fi2--------------------
        caffe_gpu_powx<Dtype>(bottom[0]->count(1),bottom_data_0 + b * step_bottom_0, float(2),m_chw);
        caffe_gpu_set(bottom[0]->count(1),Dtype(1.0),v_c_hw);
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channel_bottom_0,
                channel_bottom_0, hw_bottom_0, Dtype(1.0),
                m_chw,
                v_c_hw, Dtype(0.0),
                m_fi2);

        //fj2--------------------
        caffe_gpu_powx<Dtype>(bottom[1]->count(1),bottom_data_1 + b * step_bottom_1, float(2),m_chw);
        caffe_gpu_set(bottom[1]->count(1),Dtype(1.0),v_c_hw);
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channel_bottom_1,
                channel_bottom_1, hw_bottom_1, Dtype(1.0),
                v_c_hw,
                m_chw,Dtype(0.0),
                m_fj2);
        //compose
        caffe_gpu_axpy<Dtype>(channel_bottom_0*channel_bottom_1,Dtype(1.0),m_fi2,top_data+b*step_top);
        caffe_gpu_axpy<Dtype>(channel_bottom_0*channel_bottom_1,Dtype(1.0),m_fj2,top_data+b*step_top);
        caffe_gpu_axpy<Dtype>(channel_bottom_0*channel_bottom_1,Dtype(-2.0),m_fifj,top_data+b*step_top);

    }
    caffe_gpu_scal<Dtype>(int(top[0]->count(0)),p_beta,top_data);

    caffe_gpu_exp<Dtype>(int(top[0]->count(0)),top_data,top_data);

    /*
    cudaFree(m_fi2);
    cudaFree(m_fj2);
    cudaFree(m_fifj);
    cudaFree(m_chw);
    cudaFree(v_c_hw);
    */
}

template<typename Dtype>
void KernelRbfLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
   
    //write by gaozhi
    //ofstream out("/home/zealot/gaozhi/kernel_logupper/kernelrbf_diff.txt",ios::app);

    if ((!propagate_down[0]) && (!propagate_down[1]))
        return;
    vector<bool> pd = propagate_down;
    if (bottom[0] == bottom[1])
        pd[0] = pd[1] = true;
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff[2] = { bottom[0]->mutable_gpu_diff(), bottom[1]
            ->mutable_gpu_diff() };
    const Dtype* bottom_data[2] =
            { bottom[0]->gpu_data(), bottom[1]->gpu_data() };

    const Dtype* top_data=top[0]->gpu_data();

    const int step_top = top[0]->count(1);
    const int step_bottom[2] = { bottom[0]->count(1), bottom[1]->count(1) };

    const int hw_bottom[2]={ bottom[0]->count(2),bottom[1]->count(2)};

    const int channel_bottom[2]={bottom[0]->shape(1),bottom[1]->shape(1)};
    
    Dtype p_beta=beta_;
    /*
    Dtype *data_diff;
    Dtype *m_c_hw;
    Dtype *f_x;
    Dtype *f_y;
    */
    /*
    size_t pitch;
    CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&data_diff),
                    &pitch,sizeof(Dtype)*channel_bottom[0],channel_bottom[1]));
    CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&m_c_hw),
                    &pitch,sizeof(Dtype)*hw_bottom[0],channel_bottom[0]));
    CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&f_x),
                    &pitch,sizeof(Dtype)*hw_bottom[0],channel_bottom[0]));
    CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&f_y),
                    &pitch,sizeof(Dtype)*hw_bottom[0],channel_bottom[0]));
    */
    /*
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_diff),
                    sizeof(Dtype)*channel_bottom[0]*channel_bottom[1]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_c_hw),
                    sizeof(Dtype)*hw_bottom[0]*channel_bottom[0]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&f_x),
                    sizeof(Dtype)*hw_bottom[0]*channel_bottom[0]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&f_y),
                    sizeof(Dtype)*hw_bottom[0]*channel_bottom[0]));

    */

    Blob<Dtype> b_data_diff;
    Blob<Dtype> b_m_c_hw;
    Blob<Dtype> b_f_x;
    Blob<Dtype> b_f_y;

    b_data_diff.Reshape(1,1,channel_bottom[1],channel_bottom[0]);
    b_m_c_hw.Reshape(1,1,channel_bottom[0],hw_bottom[0]);
    b_f_x.Reshape(1,1,channel_bottom[0],hw_bottom[0]);
    b_f_y.Reshape(1,1,channel_bottom[0],hw_bottom[0]);

    Dtype * data_diff=b_data_diff.mutable_gpu_data();
    Dtype * m_c_hw=b_m_c_hw.mutable_gpu_data();
    Dtype * f_x=b_f_x.mutable_gpu_data();
    Dtype * f_y=b_f_y.mutable_gpu_data();
 
    for (int b = 0; b < bottom[0]->shape(0); ++b) {
        caffe_gpu_mul<Dtype>(channel_bottom[0]*channel_bottom[1],top_data+b*step_top,top_diff+b*step_top,data_diff);
        caffe_gpu_scal<Dtype>(channel_bottom[0]*channel_bottom[1],2*p_beta,data_diff);
        caffe_gpu_set(bottom[1]->count(1),Dtype(1.0),m_c_hw);
        //dx,fx
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channel_bottom[0],
                hw_bottom[0], channel_bottom[0], Dtype(1.0),
                data_diff,
                m_c_hw,Dtype(0.0),
                f_x);
        caffe_gpu_mul<Dtype>(bottom[0]->count(1),f_x,bottom_data[0]+b*step_bottom[0],f_x);
        //dx,fy
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channel_bottom[1],
                hw_bottom[0], channel_bottom[1], Dtype(1.0),
                data_diff,
                bottom_data[1]+b*step_bottom[1],Dtype(0.0),
                f_y);
        //dx,compose
        caffe_gpu_sub<Dtype>(bottom[0]->count(1),f_x,f_y,bottom_diff[0]+b*step_bottom[0]);
        //------------------------------------
        //dy,fy
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channel_bottom[1],
                hw_bottom[0],channel_bottom[1], Dtype(1.0),
                data_diff,
                m_c_hw,Dtype(0.0),
                f_y);
        caffe_gpu_mul<Dtype>(bottom[1]->count(1),f_y,bottom_data[1]+b*step_bottom[1],f_y);
        //dy,fx
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channel_bottom[0],
                hw_bottom[1], channel_bottom[0], Dtype(1.0),
                data_diff,
                bottom_data[0]+b*step_bottom[0],Dtype(0.0),
                f_x);
        //dy,compose
        caffe_gpu_sub<Dtype>(bottom[1]->count(1),f_y,f_x,bottom_diff[1]+b*step_bottom[1]);

        
       //int i=(rand()%(128))+1;
       //int j=(rand()%(128))+1;
        //out<<b_f_y.data_at(0,0,i,j)-b_f_x.data_at(0,0,i,j)<<endl;


    }
    /*
    cudaFree(data_diff);
    cudaFree(m_c_hw);
    cudaFree(f_x);
    cudaFree(f_y);
    */
    //out.close();
}


INSTANTIATE_LAYER_GPU_FUNCS(KernelRbfLayer);


}  // namespace caffe
