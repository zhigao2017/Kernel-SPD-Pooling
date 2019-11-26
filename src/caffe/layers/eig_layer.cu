#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/eig_layer.hpp"
#include "stdio.h"
//#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void EigLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	Forward_cpu(bottom,top);
}

template <typename Dtype>
void EigLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	if (!propagate_down[0]) {
		return;
	}

	const int mnum = bottom[0]->num();
	const int h = bottom[0]->height();
	const int dim = h*h;

	const Dtype * top_diff_0=top[0]->gpu_diff();
	const Dtype * top_diff_1=top[1]->gpu_diff();
	const Dtype * top_data_0=top[0]->gpu_data();
	const Dtype * top_data_1=top[1]->gpu_data();
	const Dtype * bottom_data=bottom[0]->gpu_data();
	Dtype * bottom_diff=bottom[0]->mutable_gpu_diff();
	/*
	for(int i=0;i<mnum;i++){
		caffe_copy(dim,top_diff_0+i*dim,bottom_diff+i*dim);
		//caffe_copy(dim,top_diff+i*dim,bottom_diff_1+i*dim);
	}
	*/

	Blob<Dtype> LeftMatU(1, 1, h, h);
	Blob<Dtype> MiddleMatS(1, 1, h, h);
	Blob<Dtype> LeftMatU_diff(1, 1, h, h);
	Blob<Dtype> MiddleMatS_diff(1, 1, h, h);

	Blob<Dtype> Identity(1, 1, h, h);
	Blob<Dtype> Unit(1, 1, h, h);
	Blob<Dtype> Sigma_i(1, 1, h, h);
	Blob<Dtype> Sigma_j(1, 1, h, h);
	Blob<Dtype> P(1, 1, h, h);

	Blob<Dtype> temp_left(1, 1, h, h);
	Blob<Dtype> temp_right(1, 1, h, h);
	Blob<Dtype> temp_eps(1, 1, h, h);

	caffe_gpu_set(h*h,Dtype(1),Identity.mutable_gpu_data());
	caffe_gpu_set(h*h,Dtype(0),Unit.mutable_gpu_data());

	Dtype eps=Dtype(pow(10,-30));
	caffe_gpu_set(dim,eps,temp_eps.mutable_gpu_data());

	for(int i=0;i<h;i++){
		caffe_gpu_set(int(1.0),Dtype(1),Unit.mutable_gpu_data()+i*h+i);
	}

	for(int i=0;i<mnum;i++){

		caffe_copy(h*h,top_data_0+i*dim,LeftMatU.mutable_gpu_data());
		caffe_copy(h*h,top_data_1+i*dim,MiddleMatS.mutable_gpu_data());
		caffe_copy(h*h,top_diff_0+i*dim,LeftMatU_diff.mutable_gpu_data());
		caffe_copy(h*h,top_diff_1+i*dim,MiddleMatS_diff.mutable_gpu_data());


		caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								MiddleMatS.gpu_data(),Identity.gpu_data(),
								Dtype(0),Sigma_i.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								Identity.gpu_data(),MiddleMatS.gpu_data(),
								Dtype(0),Sigma_j.mutable_gpu_data());

		// for P ,the paper is T, so change the order of i and j
		caffe_gpu_sub<Dtype>(h*h,Sigma_j.gpu_data(),Sigma_i.gpu_data(),P.mutable_gpu_data());
		for(int j=0;j<h;j++){
			caffe_gpu_set(int(1),Dtype(1),P.mutable_gpu_data()+j*h+j);
		}
		
		caffe_gpu_add<Dtype>(h*h,temp_eps.gpu_data(),P.gpu_data(),Sigma_i.mutable_gpu_data());
		caffe_gpu_div<Dtype>(h*h,Identity.gpu_data(),Sigma_i.gpu_data(),Sigma_j.mutable_gpu_data());
		for(int j=0;j<h;j++){
			caffe_gpu_set(int(1),Dtype(0),Sigma_j.mutable_gpu_data()+j*h+j);
		}
		//calculate the left 
		caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								LeftMatU.gpu_data(),LeftMatU_diff.gpu_data(),
								Dtype(0),temp_left.mutable_gpu_data());

		caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								temp_left.gpu_data(),Unit.gpu_data(),
								Dtype(0),Sigma_i.mutable_gpu_data());

		caffe_gpu_add<Dtype>(h*h,temp_left.gpu_data(),Sigma_i.gpu_data(),temp_right.mutable_gpu_data());
		caffe_gpu_scale<Dtype>(h*h,Dtype(0.5),temp_right.gpu_data(),temp_left.mutable_gpu_data());

		caffe_gpu_mul<Dtype>(h*h,Sigma_j.gpu_data(),temp_left.gpu_data(),temp_right.mutable_gpu_data());

		caffe_gpu_scale<Dtype>(h*h,Dtype(2.0),LeftMatU.gpu_data(),Sigma_i.mutable_gpu_data());

		caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								Sigma_i.gpu_data(),temp_right.gpu_data(),
								Dtype(0),Sigma_j.mutable_gpu_data());

		caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasTrans,h,h,h,Dtype(1.0),
								Sigma_j.gpu_data(),LeftMatU.gpu_data(),
								Dtype(0),temp_left.mutable_gpu_data());

		//calculate the right
		caffe_gpu_mul<Dtype>(h*h,Unit.gpu_data(),MiddleMatS_diff.gpu_data(),temp_right.mutable_gpu_data());

		caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								LeftMatU.gpu_data(),temp_right.gpu_data(),
								Dtype(0),Sigma_i.mutable_gpu_data());

		caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasTrans,h,h,h,Dtype(1.0),
								Sigma_i.gpu_data(),LeftMatU.gpu_data(),
								Dtype(0),temp_right.mutable_gpu_data());

		// sum the letf and right
		caffe_gpu_add<Dtype>(h*h,temp_left.gpu_data(),temp_right.gpu_data(),bottom_diff+i*dim);
		/*
		for (int k=0;k<h;k++){
			for(int l=0;l<h;l++){
				printf("%f\n", float(temp_left.data_at(0,0,k,l)+temp_right.data_at(0,0,k,l)));
			}
		}
		*/


	}
	
}

INSTANTIATE_LAYER_GPU_FUNCS(EigLayer);

}  // namespace caffe
