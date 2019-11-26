
#include <vector>

#include "caffe/layers/uppertriangle_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "stdio.h"

namespace caffe {


template<typename Dtype>
void UpperTriangleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    // write by gaozhi-------------
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
      
    const int step_bottom = bottom[0]->count(1);
    const int step_top = top[0]->count(1);

    int d =  bottom[0]->shape(2);
    int top_order;
    int bottom_order;   

    Blob<Dtype> b_matrix;
    b_matrix.Reshape(1,1,d,d);
    Dtype * matrix=b_matrix.mutable_gpu_data();

    for (int b = 0; b < bottom[0]->shape(0); ++b) {
        top_order=0;
        bottom_order=0;
        caffe_copy(d*d,bottom_data+b*step_bottom,matrix);
        caffe_gpu_scale(d*d,Dtype(sqrt(2)),matrix,matrix);
        for(int i=0;i<d;i++){

          caffe_gpu_scale(int(1.0),Dtype(sqrt(2)/2),matrix+bottom_order,matrix+bottom_order);
          caffe_copy(d-i,
                    matrix+bottom_order,
                    top_data+b*step_top+top_order);

          bottom_order=bottom_order+d+1;
          top_order=top_order+d-i;
        }
    }
}

template<typename Dtype>
void UpperTriangleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
   
    if (propagate_down[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* top_data = top[0]->gpu_data();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();


      const int step_bottom = bottom[0]->count(1);
      const int step_top = top[0]->count(1);

      int a;
      int b;

      int d =  bottom[0]->shape(2);
      int top_order;
      int bottom_order;

      Blob<Dtype> b_top_diff;
      b_top_diff.Reshape(1,1,1,top[0]->shape(1));
      Dtype * c_top_diff=b_top_diff.mutable_gpu_data();

      Blob<Dtype> b_identity;
      b_identity.Reshape(1,1,d,d);
      Dtype * identity=b_identity.mutable_gpu_data();

      Blob<Dtype> b_temp_diff;
      b_temp_diff.Reshape(1,1,d,d);
      Dtype * temp_diff=b_temp_diff.mutable_gpu_data();

      caffe_gpu_set(d*d,Dtype(0),identity);
      for(int i=0;i<d;i++){
        caffe_gpu_set(int(1.0),Dtype(1),identity+i*d+i);
      }


      for (int n = 0; n < bottom[0]->shape(0); ++n) {

        top_order=0;
        bottom_order=0;
        caffe_copy(top[0]->shape(1),top_diff+n*step_top,c_top_diff);
        caffe_gpu_scale(top[0]->shape(1),Dtype(sqrt(2)/2),c_top_diff,c_top_diff);
        caffe_gpu_set(d*d,Dtype(0.0),temp_diff);

        for(int i=0;i<d;i++){
          caffe_copy(d-i,
                    c_top_diff+top_order,
                    temp_diff+bottom_order);

          bottom_order=bottom_order+d+1;
          top_order=top_order+d-i;
        }
              
        caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,d,d,d,Dtype(1.0),temp_diff,identity,Dtype(1.0),temp_diff);

        for(int i=0;i<d;i++){
          caffe_gpu_scale(int(1.0),Dtype(sqrt(2)/2),temp_diff+i*d+i,temp_diff+i*d+i);
        } 

        caffe_copy(d*d,temp_diff,bottom_diff+n*step_bottom);

      }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(UpperTriangleLayer);


}  // namespace caffe
