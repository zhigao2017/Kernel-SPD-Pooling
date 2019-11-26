#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cluster_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using namespace std;

template <typename Dtype>
void ClusterLossLayer<Dtype>::Compute_distance_data_gpu(const int M, const int K, const Dtype* bottom,
        const Dtype* label, const Dtype* center, Dtype* distance) {
  for(int i=0;i < M;i++){
    Dtype label_value;
    caffe_copy(int(1),label+i,&label_value);
    caffe_gpu_sub<Dtype>(K,bottom+i*K,center+int(label_value)*K,distance+i*K);
  }
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::Compute_center_diff_gpu(const int M, const int K, const int N,
        const Dtype* label, const Dtype* distance, const Dtype* center_distance, Dtype* variation_sum, 
        Dtype* center_diff, Dtype* center_among_diff) {
  caffe_gpu_set(N*K,Dtype(0),center_among_diff);
  for(int i=0;i<N;i++){
    int count = 0;
    for (int m = 0; m < M; m++) {
      Dtype label_value;
      caffe_copy(int(1),label+m,&label_value);
      if (label_value == i) {
        count++;
        caffe_gpu_sub<Dtype>(K,variation_sum+i*K,distance+m*K,variation_sum+i*K);
      }
    }
    caffe_copy<Dtype>(K,variation_sum+i * K,center_diff+i * K);
    caffe_gpu_scal<Dtype>(K,Dtype(1)/(count + (Dtype)1.),center_diff+i * K);
    for(int k=0;k<N;k++){
      caffe_gpu_sub<Dtype>(K,center_among_diff+i*K,center_distance+i*N*K+k*K,center_among_diff+i*K);
    }
  } 
  caffe_gpu_scal<Dtype>(N*K,Dtype(1)/(N-1),center_among_diff);
  caffe_gpu_add<Dtype>(N*K,center_among_diff,center_diff,center_diff);
}


template <typename Dtype>
void ClusterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 
  Compute_distance_data_gpu(M_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                            this->blobs_[0]->gpu_data(), distance_.mutable_gpu_data());

  const Dtype* center = this->blobs_[0]->gpu_data();
  Dtype* center_distance_data = center_distance_.mutable_gpu_data();
  Dtype* variation_sum_data = variation_sum_.mutable_gpu_data();
  for (int i=0;i<N_;i++){
    for(int j=0;j<N_;j++){
      caffe_gpu_sub<Dtype>(K_,center + i * K_,center + j * K_,center_distance_data+i*N_*K_+j*K_);
    }
  }

  Dtype dot;
  Dtype dot_center;
  caffe_gpu_dot(M_ * K_, distance_.gpu_data(), distance_.gpu_data(), &dot);
  caffe_gpu_dot(N_ *N_* K_, center_distance_.gpu_data(), center_distance_.gpu_data(), &dot_center);
  Dtype loss = dot / M_/Dtype(2) - dot_center / (N_-1)/(N_-1)/Dtype(2);

  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int nthreads = N_;
  caffe_gpu_set(N_ * K_, (Dtype)0., variation_sum_.mutable_gpu_data());
  Dtype * center_among_diff=center_among.mutable_gpu_data();

  Compute_center_diff_gpu( M_, K_,N_, bottom[1]->gpu_data(), distance_.gpu_data(),
                                center_distance_.mutable_gpu_data(), 
                                variation_sum_.mutable_gpu_data(), this->blobs_[0]->mutable_gpu_diff(),
                                center_among_diff);      
  
  //gradient respect to x
  if (propagate_down[0]) {
    //caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / M_, 
                             //distance_.gpu_data(), bottom[0]->mutable_gpu_diff());
    caffe_gpu_scale(M_ * K_, Dtype(1) / M_, 
                             distance_.gpu_data(), bottom[0]->mutable_gpu_diff());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ClusterLossLayer);

}  // namespace caffe