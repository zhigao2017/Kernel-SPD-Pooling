#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/kernelmap_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KernelMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* weight = this->blobs_[0]->mutable_gpu_data();

    Blob<Dtype> b_temp;
    b_temp.Reshape(1,1,N_,M_);
    Dtype * temp=b_temp.mutable_gpu_data();

    const int step_bottom = bottom[0]->count(1);
    const int step_top = top[0]->count(1);

    for (int b = 0; b < bottom[0]->shape(0); ++b) {
      caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,
                            N_,M_,M_,Dtype(1.0),
                            weight,bottom_data+b*step_bottom,Dtype(0.0),
                            temp);

      caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                            N_,N_,M_,Dtype(1.0),
                            temp,weight,Dtype(0.0),
                            top_data+b*step_top);
    }

}

template <typename Dtype>
void KernelMapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    if (this->param_propagate_down_[0]) {
        Backward_cpu(top,propagate_down,bottom);
    }
}


INSTANTIATE_LAYER_GPU_FUNCS(KernelMapLayer);

}  // namespace caffe
