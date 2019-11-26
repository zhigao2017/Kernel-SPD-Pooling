#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/kernelpv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KernelPvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();

    //Dtype *temp=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * D1_ * D2_));
    //Dtype *temp_one=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * int(1.0) * D1_));
    Blob<Dtype> b_temp;
    Blob<Dtype> b_temp_one;

    b_temp.Reshape(1,1,D1_,D2_);
    b_temp_one.Reshape(1,1,1,D1_);

    Dtype * temp=b_temp.mutable_gpu_data();
    Dtype * temp_one=b_temp_one.mutable_gpu_data();

    const int step_bottom = bottom[0]->count(1);
    const int step_top = top[0]->count(1);

    caffe_gpu_set(int(1.0) * D1_,Dtype(1.0),temp_one);

    for (int b = 0; b < bottom[0]->shape(0); ++b) {

      caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                            D1_,D2_,D1_,Dtype(1.0),
                            bottom_data+b*step_bottom,weight,Dtype(0.0),
                            temp);

      caffe_gpu_mul<Dtype>(D1_*D2_,weight,temp,temp);


      caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                            int(1.0),D2_,D1_,Dtype(1.0),
                            temp_one,temp,Dtype(0.0),
                            top_data+b*step_top);
    }

}

template <typename Dtype>
void KernelPvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

   if (this->param_propagate_down_[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* top_data = top[0]->gpu_data();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
      const Dtype* weight = this->blobs_[0]->gpu_data();

      const int step_bottom = bottom[0]->count(1);
      const int step_top = top[0]->count(1);

      //Dtype *temp_w1=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * D1_ * D2_));
      //Dtype *temp_w2=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * D1_ * D2_));
      //Dtype *temp_s=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * D1_ * D2_));
      //Dtype *temp_one=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * D1_ * int(1.0)));
      //Dtype *temp_y=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * D1_ * D2_));

      Blob<Dtype> b_temp_w1;
      Blob<Dtype> b_temp_w2;
      Blob<Dtype> b_temp_s;
      Blob<Dtype> b_temp_one;
      Blob<Dtype> b_temp_y;

      b_temp_w1.Reshape(1,1,D1_,D2_);
      b_temp_w2.Reshape(1,1,D1_,D2_);
      b_temp_s.Reshape(1,1,D1_,D2_);
      b_temp_one.Reshape(1,1,D1_,int(1.0));
      b_temp_y.Reshape(1,1,D1_,D2_);

      Dtype * temp_w1=b_temp_w1.mutable_gpu_data();
      Dtype * temp_w2=b_temp_w2.mutable_gpu_data();
      Dtype * temp_s=b_temp_s.mutable_gpu_data();
      Dtype * temp_one=b_temp_one.mutable_gpu_data();
      Dtype * temp_y=b_temp_y.mutable_gpu_data();

      caffe_gpu_set(D1_*int(1.0),Dtype(1.0),temp_one);

      for (int b = 0; b < bottom[0]->shape(0); ++b) {
       
        caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                              D1_,D2_,int(1.0),Dtype(1.0),
                              temp_one,top_diff+b*step_top,Dtype(0.0),
                              temp_s);
        // Gradient with respect to bottom data
        if (propagate_down[0]){

          caffe_gpu_mul<Dtype>(D1_*D2_,temp_s,weight,temp_y);

          caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasTrans,
                              D1_,D1_,D2_,Dtype(1.0),
                              temp_y,weight,Dtype(0.0),
                              bottom_diff+b*step_bottom);
        }
        // Gradient with respect to weight
        if (this->param_propagate_down_[0]){
          caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                              D1_,D2_,D1_,Dtype(1.0),
                              bottom_data+b*step_bottom,weight,Dtype(0.0),
                              temp_w1);
          caffe_gpu_mul<Dtype>(D1_*D2_,temp_s,temp_w1,temp_w1);

          caffe_gpu_mul<Dtype>(D1_*D2_,temp_s,weight,temp_w2);
          caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,
                              D1_,D2_,D1_,Dtype(1.0),
                              bottom_data+b*step_bottom,temp_w2,Dtype(0.0),
                              temp_w2);

          
          if (b==0)
            caffe_gpu_set(D1_*D2_,Dtype(0.0),weight_diff);
          caffe_gpu_add<Dtype>(D1_*D2_,weight_diff,temp_w1,weight_diff);
          caffe_gpu_add<Dtype>(D1_*D2_,weight_diff,temp_w2,weight_diff);

          
        }
      }
    }
}


INSTANTIATE_LAYER_GPU_FUNCS(KernelPvLayer);

}  // namespace caffe
