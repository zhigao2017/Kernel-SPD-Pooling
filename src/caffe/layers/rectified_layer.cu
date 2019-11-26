#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/rectified_layer.hpp"

//#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void RectifiedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	Forward_cpu(bottom,top);
}

template <typename Dtype>
void RectifiedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	if ((!propagate_down[0]) && (!propagate_down[1])) {
		return;
	}

	const Dtype* bottom_data_0 = bottom[0]->gpu_data();
	const Dtype* bottom_data_1 = bottom[1]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff_0 = bottom[0]->mutable_gpu_diff(); 
	Dtype* bottom_diff_1 = bottom[1]->mutable_gpu_diff(); 

	const int mnum = bottom[0]->num();
	const int h = bottom[0]->height();
	const int dim = h*h;

	Blob<Dtype> LeftMatU(1, 1, h, h);
	Blob<Dtype> MiddleMatS(1, 1, h, h);
	Blob<Dtype> Unit(1, 1, h, h);
	Blob<Dtype> temp_1(1, 1, h, h);
	Blob<Dtype> temp_2(1, 1, h, h);
	Blob<Dtype> temp_log(1, 1, h, h);


	caffe_gpu_set(h*h,Dtype(0),Unit.mutable_gpu_data());
	for(int i=0;i<h;i++){
		caffe_gpu_set(int(1.0),Dtype(1),Unit.mutable_gpu_data()+i*h+i);
	}

	for(int i=0;i<mnum;i++){

		caffe_copy(h*h,bottom_data_0+i*dim,LeftMatU.mutable_gpu_data());
		caffe_copy(h*h,bottom_data_1+i*dim,MiddleMatS.mutable_gpu_data());

		//the gradient of U
		caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								top_diff+i*dim,Unit.gpu_data(),
								Dtype(0),temp_1.mutable_gpu_data());
		caffe_gpu_add<Dtype>(h*h,top_diff+i*dim,temp_1.gpu_data(),temp_2.mutable_gpu_data());
		caffe_gpu_scale(h*h,Dtype(0.5),temp_2.gpu_data(),temp_1.mutable_gpu_data());
		caffe_gpu_scale(h*h,Dtype(2),temp_1.gpu_data(),temp_2.mutable_gpu_data());

		caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								temp_2.gpu_data(),LeftMatU.gpu_data(),
								Dtype(0),temp_1.mutable_gpu_data());

		caffe_gpu_set(h*h,Dtype(0),temp_log.mutable_gpu_data());

		for(int j = 0; j < h; j ++)
		{
			Dtype tmp = MiddleMatS.data_at(0,0,j,j) > sigma_ ?  MiddleMatS.data_at(0,0,j,j) : sigma_;
			caffe_gpu_set(1, tmp, temp_log.mutable_gpu_data()+j*h+j);
		}

		caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								temp_1.gpu_data(),temp_log.gpu_data(),
								Dtype(0),bottom_diff_0+i*dim);

		//the gradient of sigma
		caffe_gpu_set(h*h, Dtype(0), temp_1.mutable_gpu_data());
		for(int j=0;j<h;j++)
		{
			Dtype tmp = MiddleMatS.data_at(0,0,j,j) > sigma_ ?  1 : 0;
			caffe_gpu_set(1,tmp,temp_1.mutable_gpu_data()+j*h+j);
		}
		caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasTrans,h,h,h,Dtype(1.0),
								temp_1.gpu_data(),LeftMatU.gpu_data(),
								Dtype(0),temp_2.mutable_gpu_data());

		caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								top_diff+i*dim,Unit.gpu_data(),
								Dtype(0),temp_log.mutable_gpu_data());
		caffe_gpu_add<Dtype>(h*h,top_diff+i*dim,temp_log.gpu_data(),temp_1.mutable_gpu_data());

		caffe_gpu_scale(h*h,Dtype(0.5),temp_1.gpu_data(),temp_log.mutable_gpu_data());

		caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								temp_2.gpu_data(),temp_log.gpu_data(),
								Dtype(0),temp_1.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								temp_1.gpu_data(),LeftMatU.gpu_data(),
								Dtype(0),bottom_diff_1+i*dim);
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(RectifiedLayer);

}  // namespace caffe
