#ifndef CAFFE_CLUSTER_LOSS_LAYER_HPP_
#define CAFFE_CLUSTER_LOSS_LAYER_HPP_
#include <iostream>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class ClusterLossLayer : public LossLayer<Dtype> {
 public:
  explicit ClusterLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ClusterLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return -1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  
  Blob<Dtype> distance_;
  Blob<Dtype> variation_sum_;
  Blob<Dtype> center_distance_;
  Blob<Dtype> center_among;

  void Compute_distance_data_gpu(const int M, const int K, const Dtype* bottom,
        const Dtype* label, const Dtype* center, Dtype* distance);
  void Compute_center_diff_gpu( const int M, const int K, const int N,
        const Dtype* label, const Dtype* distance, const Dtype* center_distance, Dtype* variation_sum, 
        Dtype* center_diff, Dtype* center_among_diff);

};

}  // namespace caffe

#endif  // ClusterLossLayer