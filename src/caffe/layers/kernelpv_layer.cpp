#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/kernelpv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KernelPvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 	const KernelPvParameter& kernelpv_param = this->layer_param_.kernelpv_param();
    D2_=kernelpv_param.out_num();
    D1_=bottom[0]->shape(2);
    vector<int> weight_shape(2);
    weight_shape[0] = D1_;
    weight_shape[1] = D2_;
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    //fill the weight filler
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.kernelpv_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    //parameter initialization
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void KernelPvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 	int h=bottom[0]->shape(2);
 	int w=bottom[0]->shape(3);
 	CHECK_EQ(h, w) << "The width and higth must be equal";

 	vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(2);
  top_shape[1]=D2_;
 	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void KernelPvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
	  const Dtype* weight = this->blobs_[0]->cpu_data();

	  Dtype *temp=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * D1_ * D2_));
    Dtype *temp_one=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * int(1.0) * D1_));

    const int step_bottom = bottom[0]->count(1);
    const int step_top = top[0]->count(1);

    caffe_set(int(1.0) * D1_,Dtype(1.0),temp_one);

  	for (int b = 0; b < bottom[0]->shape(0); ++b) {

  		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                            D1_,D2_,D1_,Dtype(1.0),
                            bottom_data+b*step_bottom,weight,Dtype(0.0),
                            temp);

      caffe_mul<Dtype>(D1_*D2_,weight,temp,temp);


  		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                            int(1.0),D2_,D1_,Dtype(1.0),
                            temp_one,temp,Dtype(0.0),
                            top_data+b*step_top);
  	}

  	free(temp);
    free(temp_one);
}

template <typename Dtype>
void KernelPvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    if (this->param_propagate_down_[0]) {
      const Dtype* top_diff = top[0]->cpu_diff();
      const Dtype* top_data = top[0]->cpu_data();
      const Dtype* bottom_data = bottom[0]->cpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
      const Dtype* weight = this->blobs_[0]->cpu_data();

      const int step_bottom = bottom[0]->count(1);
      const int step_top = top[0]->count(1);

      Dtype *temp_w1=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * D1_ * D2_));
      Dtype *temp_w2=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * D1_ * D2_));
      Dtype *temp_s=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * D1_ * D2_));
      Dtype *temp_one=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * D1_ * int(1.0)));
      Dtype *temp_y=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * D1_ * D2_));

      caffe_set(D1_*int(1.0),Dtype(1.0),temp_one);

      for (int b = 0; b < bottom[0]->shape(0); ++b) {
       
        caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                              D1_,D2_,int(1.0),Dtype(1.0),
                              temp_one,top_diff+b*step_top,Dtype(0.0),
                              temp_s);
        // Gradient with respect to bottom data
        if (propagate_down[0]){

          caffe_mul<Dtype>(D1_*D2_,temp_s,weight,temp_y);

          caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasTrans,
                              D1_,D1_,D2_,Dtype(1.0),
                              temp_y,weight,Dtype(0.0),
                              bottom_diff+b*step_bottom);
        }
        // Gradient with respect to weight
        if (this->param_propagate_down_[0]){
          caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                              D1_,D2_,D1_,Dtype(1.0),
                              bottom_data+b*step_bottom,weight,Dtype(0.0),
                              temp_w1);
          caffe_mul<Dtype>(D1_*D2_,temp_s,temp_w1,temp_w1);

          caffe_mul<Dtype>(D1_*D2_,temp_s,weight,temp_w2);
          caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,
                              D1_,D2_,D1_,Dtype(1.0),
                              bottom_data+b*step_bottom,temp_w2,Dtype(0.0),
                              temp_w2);

          
          if (b==0)
            caffe_set(D1_*D2_,Dtype(0.0),weight_diff);
          caffe_add<Dtype>(D1_*D2_,weight_diff,temp_w1,weight_diff);
          caffe_add<Dtype>(D1_*D2_,weight_diff,temp_w2,weight_diff);

        }
      }
      free(temp_s);
      free(temp_w1);
      free(temp_w2);
      free(temp_one);
      free(temp_y);
    }
}

#ifdef CPU_ONLY
STUB_GPU(KernelPvLayer);
#endif

INSTANTIATE_CLASS(KernelPvLayer);
REGISTER_LAYER_CLASS(KernelPv);

}  // namespace caffe
