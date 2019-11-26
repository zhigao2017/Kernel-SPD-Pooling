#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/uppertriangle_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void UpperTriangleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void UpperTriangleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 	int h=bottom[0]->shape(2);
 	int w=bottom[0]->shape(3);
 	CHECK_EQ(h, w) << "The width and higth must be equal";

 	vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(2);
  top_shape[1]=(h*h+h)/2;
 	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void UpperTriangleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
	  
    const int step_bottom = bottom[0]->count(1);
    const int step_top = top[0]->count(1);

    int d =  bottom[0]->shape(2);
    int top_order;
    int bottom_order;   

    Dtype * matrix=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * d * d));

    for (int b = 0; b < bottom[0]->shape(0); ++b) {
        top_order=0;
        bottom_order=0;
        caffe_copy(d*d,bottom_data+b*step_bottom,matrix);
        caffe_cpu_scale(d*d,Dtype(1.4142135623730950488016887242097),matrix,matrix);
        for(int i=0;i<d;i++){

          caffe_cpu_scale(int(1.0),Dtype(0.70710678118655),matrix+bottom_order,matrix+bottom_order);
          caffe_copy(d-i,
                    matrix+bottom_order,
                    top_data+b*step_top+top_order);

          bottom_order=bottom_order+d+i+1;
          top_order=top_order+d-i;
        }
    }
    free(matrix);
}

template <typename Dtype>
void UpperTriangleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    if (this->param_propagate_down_[0]) {
      const Dtype* top_diff = top[0]->cpu_diff();
      const Dtype* top_data = top[0]->cpu_data();
      const Dtype* bottom_data = bottom[0]->cpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();


      const int step_bottom = bottom[0]->count(1);
      const int step_top = top[0]->count(1);

      int a;
      int b;

      Dtype element_diff;

      int d =  bottom[0]->shape(2);
      for (int b = 0; b < bottom[0]->shape(0); ++b) {
          for(int i=0;i<d;i++){
             for(int j=0;j<d;j++){
                if(i<=j) {a=i;b=j;}
                else {a=j;b=i;}

                element_diff=*(top_diff+b*step_top+
                              (d+d-a+1)*(a)/2+b);
                if(a!=b) element_diff=element_diff*Dtype(0.70710678118655);

                caffe_copy(int(1.0),&element_diff,
                          bottom_diff+b*step_bottom+d*i+j);
             }
          }

      }
    }

}

#ifdef CPU_ONLY
STUB_GPU(UpperTriangleLayer);
#endif

INSTANTIATE_CLASS(UpperTriangleLayer);
REGISTER_LAYER_CLASS(UpperTriangle);

}  // namespace caffe
