#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/kernelmap_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KernelMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 	const KernelMapParameter& kernelmap_param = this->layer_param_.kernelmap_param();
    N_=kernelmap_param.out_num();
    M_=bottom[0]->shape(2);
    LR_=kernelmap_param.layer_lr();
    vector<int> weight_shape(2);
    weight_shape[0] = M_;
    weight_shape[1] = N_;
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    //fill the weight filler
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.kernelmap_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    //parameter initialization
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void KernelMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 	int h=bottom[0]->shape(2);
 	int w=bottom[0]->shape(3);
 	CHECK_EQ(h, w) << "The width and higth must be equal";
 	vector<int> top_shape = bottom[0]->shape();
 	top_shape[2]=N_;
 	top_shape[3]=N_;
 	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void KernelMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
	  const Dtype* weight = this->blobs_[0]->mutable_cpu_data();


	  Dtype *temp=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * N_ * M_));
    const int step_bottom = bottom[0]->count(1);
    const int step_top = top[0]->count(1);

  	for (int b = 0; b < bottom[0]->shape(0); ++b) {
  		caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,
                            N_,M_,M_,Dtype(1.0),
                            weight,bottom_data+b*step_bottom,Dtype(0.0),
                            temp);

  		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                            N_,N_,M_,Dtype(1.0),
                            temp,weight,Dtype(0.0),
                            top_data+b*step_top);
  	}

  	free(temp);
}

template <typename Dtype>
void KernelMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    if (this->param_propagate_down_[0]) {
      const Dtype* top_diff = top[0]->cpu_diff();
      const Dtype* top_data = top[0]->cpu_data();
      const Dtype* bottom_data = bottom[0]->cpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
      Dtype* weight = this->blobs_[0]->mutable_cpu_data();

      const int step_bottom = bottom[0]->count(1);
      const int step_top = top[0]->count(1);
      Dtype *temp_x=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * M_ * N_));
      Dtype *temp_w=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * M_ * N_));
      Dtype *temp_y=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * M_ * N_));
      Dtype *temp_b1=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * M_ * M_));
      Dtype *temp_b2=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * M_ * M_));
      Dtype *temp_b3=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * M_ * M_));
      Dtype *gra_euclidean=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * M_ * N_));
      for (int b = 0; b < bottom[0]->shape(0); ++b) {
        // Gradient with respect to bottom data
        if (propagate_down[0]){
          caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                              M_,N_,N_,Dtype(1.0),
                              weight,top_diff+b*step_top,Dtype(0.0),
                              temp_x);
          caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasTrans,
                              M_,M_,N_,Dtype(1.0),
                              temp_x,weight,Dtype(0.0),
                              bottom_diff+b*step_bottom);
        }
        // Gradient with respect to weight
        if (this->param_propagate_down_[0]){

          caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,
                              M_,N_,M_,Dtype(1.0),
                              bottom_data+b*step_bottom,weight,Dtype(0.0),
                              temp_w);
          caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                              M_,N_,N_,Dtype(1.0),
                              temp_w,top_diff+b*step_top,Dtype(0.0),
                              gra_euclidean);
          caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                              M_,N_,M_,Dtype(1.0),
                              bottom_data+b*step_bottom,weight,Dtype(0.0),
                              temp_w);
          caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasTrans,
                              M_,N_,N_,Dtype(1.0),
                              temp_w,top_diff+b*step_top,Dtype(1.0),
                              gra_euclidean);

          if (b==0)
            caffe_set(M_*N_,Dtype(0.0),temp_y);
          caffe_add<Dtype>(M_*N_,temp_y,gra_euclidean,temp_y);
        }
      }

      /*
      caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasTrans,
                          M_,M_,N_,Dtype(1.0),
                          weight,weight,Dtype(0.0),
                          temp_b1);

      caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                          M_,N_,M_,Dtype(1.0),
                          temp_b1,weight,Dtype(0.0),
                          temp_w);
      */
/*
      caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasTrans,
                          M_,M_,N_,Dtype(1.0),
                          weight,temp_y,Dtype(0.0),
                          temp_b1);

      caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                          M_,N_,M_,Dtype(1.0),
                          temp_b1,weight,Dtype(0.0),
                          temp_w);
      */

      caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,
                          M_,M_,N_,Dtype(1.0),
                          weight,temp_y,Dtype(0.0),
                          temp_b1);

      caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,
                          M_,M_,N_,Dtype(1.0),
                          temp_y,weight,Dtype(0.0),
                          temp_b2);

      caffe_add<Dtype>(M_*N_,temp_b1,temp_b2,temp_b3);

      caffe_scal<Dtype>(M_*N_, Dtype(0.5), temp_b3);

      caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                          M_,N_,M_,Dtype(1.0),
                          weight,temp_b3,Dtype(0.0),
                          temp_w);

      caffe_sub<Dtype>(M_*N_,temp_y,temp_w,gra_euclidean);

      Dtype *temp_weight=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * M_ * N_));
      caffe_copy(M_*N_,weight,temp_weight);
      caffe_axpy<Dtype>(M_ * N_,int(-1.0)*LR_,gra_euclidean,temp_weight);
      caffe_cpu_qrq<Dtype>(M_,N_,temp_weight);
      caffe_copy(M_*N_,temp_weight,weight);

      caffe_set(M_*N_,Dtype(0.0),weight_diff);
      free(temp_x);
      free(temp_w);
      free(gra_euclidean);
      free(temp_weight);
      free(temp_b1);
      free(temp_b2);
      free(temp_b3);
    }
}

#ifdef CPU_ONLY
STUB_GPU(KernelMapLayer);
#endif

INSTANTIATE_CLASS(KernelMapLayer);
REGISTER_LAYER_CLASS(KernelMap);

}  // namespace caffe
