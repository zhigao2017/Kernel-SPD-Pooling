#include <vector>
#include <iostream>

#include "caffe/layers/rectified_layer.hpp"
#include "caffe/util/math_functions.hpp"

//#include <algorithm>
//#include <cfloat>
//#include <math.h>

namespace caffe {

using std::min;
using std::max;
using std::cout;

template <typename Dtype>
void RectifiedLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), 2) << "Rectified Layer takes a single blob as input.";
	CHECK_EQ(top.size(), 1) << "Rectified Layer takes a single blob as output.";
	CHECK_EQ(bottom[0]->channels(), 1) << "Rectified Layer inputs 2D matrices.";
	CHECK_EQ(bottom[0]->height(), bottom[0]->width()) << "Rectified Layer inputs square 2D matrices.";
	CHECK_EQ(bottom[1]->channels(), 1) << "Rectified Layer inputs 2D matrices.";
	CHECK_EQ(bottom[1]->height(), bottom[1]->width()) << "Rectified Layer inputs square 2D matrices.";
	CHECK_EQ(bottom[0]->height(), bottom[1]->width()) << "Rectified Layer inputs sigma must equal to u.";

	const RectifiedParameter& rectified_param = this->layer_param_.rectified_param();
    sigma_=rectified_param.sigma();
}

template <typename Dtype>
void RectifiedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	 top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
	      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void RectifiedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data_0 = bottom[0]->cpu_data();
	const Dtype* bottom_data_1 = bottom[1]->cpu_data();

	Dtype* top_data = top[0]->mutable_cpu_data();

	const int mnum = bottom[0]->num();
	const int h = bottom[0]->height();
	const int dim = h*h;

	Blob<Dtype> LeftMatU(1, 1, h, h);
	Blob<Dtype> MiddleMatS(1, 1, h, h);
	Blob<Dtype> InterMat(1,1,h,h);

	for(int i = 0; i < mnum; i ++)
	{

		caffe_copy(h*h,bottom_data_0+i*dim,LeftMatU.mutable_cpu_data());
		caffe_copy(h*h,bottom_data_1+i*dim,MiddleMatS.mutable_cpu_data());

		int ccount = 0;
		for(int j = 0; j < h; j ++)
		{
			//if( MiddleMatS.data_at(0,0,j,j)<=0 )
			//{
				//ccount ++;
			//}
			Dtype tmp = MiddleMatS.data_at(0,0,j,j) > sigma_ ?  MiddleMatS.data_at(0,0,j,j) : sigma_;
			//Dtype tmp = Dtype(log( MiddleMatS.data_at(0,0,j,j) ));
			caffe_set(1,tmp, MiddleMatS.mutable_cpu_data()+j*h+j);
			//std::cout<< tmp<<std::endl;
		}
		//std::cout << ccount << " negative sigular elements in total." << std::endl;

		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, h, h, h, Dtype(1.), 
				LeftMatU.cpu_data(), MiddleMatS.cpu_data(), Dtype(0.), InterMat.mutable_cpu_data());
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, h, h, h, Dtype(1.), 
				InterMat.cpu_data(), LeftMatU.cpu_data(), Dtype(0.), top_data+i*dim);
	}

	//caffe_copy(bottom[0]->count(), bottom_data, top_data);

	
/*
	const int mnum = bottom[0]->num();
	const int dim = bottom[0]->width();
	const int odim = bottom[0]->count() / bottom[0]->num();

	//memset(top_data, 0, sizeof(Dtype) * odim * vidnum);
	caffe_set(bottom[0]->count(), Dtype(0), top_data);

	for(int i = 0; i < mnum; i ++)
	{
		const Dtype* bdata = bottom_data + i*odim;
		Dtype* tdata = top_data + i*odim;

		for(int j = 0; j < dim; j++)
		{
			//(tdata+sizeof(Dtype)*(j*dim+j)) = 1;
			caffe_set(1, Dtype(1.), tdata+j*dim+j);
		}

		caffe_cpu_invpb('N', dim, dim, bdata, tdata);
	}*/
}

template <typename Dtype>
void RectifiedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if ((!propagate_down[0]) && (!propagate_down[1])) {
		return;
	}

	const Dtype* bottom_data_0 = bottom[0]->cpu_data();
	const Dtype* bottom_data_1 = bottom[1]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff_0 = bottom[0]->mutable_cpu_diff(); 
	Dtype* bottom_diff_1 = bottom[1]->mutable_cpu_diff(); 

	const int mnum = bottom[0]->num();
	const int h = bottom[0]->height();
	const int dim = h*h;

	//Blob<Dtype> LeftMatU(1, 1, h, h);
	//Blob<Dtype> MiddleMatS(1, 1, h, h);
	//Blob<Dtype> Unit(1, 1, h, h);
	//Blob<Dtype> temp_1(1, 1, h, h);
	//Blob<Dtype> temp_2(1, 1, h, h);
	//Blob<Dtype> temp_log(1, 1, h, h);

	Dtype *LeftMatU=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *MiddleMatS=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *Unit=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *temp_1=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *temp_2=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *temp_log=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));



	caffe_set(h*h,Dtype(0),Unit);
	for(int i=0;i<h;i++){
		caffe_set(int(1.0),Dtype(1),Unit+i*h+i);
	}

	for(int i=0;i<mnum;i++){

		caffe_copy(h*h,bottom_data_0+i*dim,LeftMatU);
		caffe_copy(h*h,bottom_data_1+i*dim,MiddleMatS);

		//the gradient of U
		caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								top_diff+i*dim,Unit,
								Dtype(0),temp_1);
		caffe_add<Dtype>(h*h,top_diff+i*dim,temp_1,temp_2);
		caffe_cpu_scale(h*h,Dtype(0.5),temp_2,temp_1);
		caffe_cpu_scale(h*h,Dtype(2),temp_1,temp_2);

		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								temp_2,LeftMatU,
								Dtype(0),temp_1);

		caffe_set(h*h,Dtype(0),temp_log);
		for(int j = 0; j < h; j ++)
		{
			Dtype tmp = (*(MiddleMatS+j*h+j)) > 0 ? Dtype(log(*(MiddleMatS+j*h+j))) : 0;
			caffe_set(1, tmp, temp_log+j*h+j);
		}

		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								temp_1,temp_log,
								Dtype(0),bottom_diff_0+i*dim);

		//the gradient of sigma
		//caffe_div<Dtype>(h*h,Unit.cpu_data(),MiddleMatS.cpu_data(),temp.mutable_cpu_data());
		caffe_set(h*h, Dtype(0), temp_1);
		for(int j=0;j<h;j++)
		{
			Dtype tmp = (*(MiddleMatS+j*h+j)) > 0 ? Dtype(1/(*(MiddleMatS+j*h+j))) : 0;
			//Dtype tmp = 1/MiddleMatS.data_at(0,0,j,j) ;
			caffe_set(1,tmp,temp_1+j*h+j);
		}
		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasTrans,h,h,h,Dtype(1.0),
								temp_1,LeftMatU,
								Dtype(0),temp_2);

		caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								top_diff+i*dim,Unit,
								Dtype(0),temp_log);
		caffe_add<Dtype>(h*h,top_diff+i*dim,temp_log,temp_1);
		caffe_cpu_scale(h*h,Dtype(0.5),temp_1,temp_log);

		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								temp_2,temp_log,
								Dtype(0),temp_1);
		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								temp_1,LeftMatU,
								Dtype(0),bottom_diff_1+i*dim);
	}

	free(LeftMatU);
	free(MiddleMatS);
	free(Unit);
	
	free(temp_1);
	free(temp_2);
	free(temp_log);
}


#ifdef CPU_ONLY
STUB_GPU(RectifiedLayer);
#endif

INSTANTIATE_CLASS(RectifiedLayer);
REGISTER_LAYER_CLASS(Rectified);

}  // namespace caffe
