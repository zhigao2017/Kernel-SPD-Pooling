#include <vector>
#include <iostream>

#include "caffe/layers/eig_layer.hpp"
#include "caffe/util/math_functions.hpp"

//#include <algorithm>
//#include <cfloat>
//#include <math.h>

namespace caffe {

using namespace std;

template <typename Dtype>
void EigLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), 1) << "Eig Layer takes a single blob as input.";
	CHECK_EQ(top.size(), 2) << "Eig Layer takes two blobs as output.";
	CHECK_EQ(bottom[0]->channels(), 1) << "Eig Layer inputs 2D matrices.";
	CHECK_EQ(bottom[0]->height(), bottom[0]->width()) << "Eig Layer inputs square 2D matrices.";
}

template <typename Dtype>
void EigLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	 //top.resize(2);
	 //first is U ,and the second is sigma
	 top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
	      bottom[0]->height(), bottom[0]->width());
	 top[1]->Reshape(bottom[0]->num(), bottom[0]->channels(),
	      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void EigLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->cpu_data();

	Dtype* top_data_0 = top[0]->mutable_cpu_data();
	Dtype* top_data_1 = top[1]->mutable_cpu_data();

	const int step_top_0=top[0]->count(1);
	const int step_top_1=top[1]->count(1);

	const int mnum = bottom[0]->num();
	const int h = bottom[0]->height();
	const int dim = h*h;

	Blob<Dtype> LeftMatU(1, 1, h, h);
	Blob<Dtype> MiddleVecS(1, 1, h, 1);
	Blob<Dtype> MiddleMatS(1, 1, h, h);

	for(int i = 0; i < mnum; i ++)
	{
		const Dtype* bdata = bottom_data + i*dim;
		//Dtype* tdata = top_data + i*dim;

		/*for(int j = 0; j < h; j++)
                {
                        for(int k = 0; k<h; k++)
                        {
                               std::cout << bottom[0]->data_at(i,0,j,k) << " ";
                        }
                        std::cout << std::endl;
                }
		std::cout << std::endl;
		for(int j = 0; j < h; j++)
                {
                        for(int k = 0; k<h; k++)
                        {
                               std::cout << *(bdata+j*h+k) << " ";
                        }
                        std::cout << std::endl;
                }
		std::cout << std::endl;*/
		caffe_cpu_svd(h, h, bdata, LeftMatU.mutable_cpu_data(), MiddleVecS.mutable_cpu_data());
                /*for(int j = 0; j < h; j++)
                {
               		 std::cout << MiddleVecS.data_at(0,0,j,0) << " ";
                }
		std::cout << std::endl;*/
        int ccount = 0;
        int nmin=0;
        int ngap=0;
		caffe_set(dim, Dtype(0.), MiddleMatS.mutable_cpu_data());
		for(int j = 0; j < h; j ++)
		{
			Dtype tmp = MiddleVecS.data_at(0,0,j,0);
			/*
			if(tmp==MiddleMatS.data_at(0,0,j-1,j-1))
			{

			}

			if((tmp==0)&&(ccount==0))
			{
				ccount=1;
				ngap=h-j+1;
				//if(j==0){nmin=1/ngap;}
				nmin=MiddleVecS.data_at(0,0,j-1,0)/ngap;
			}
			if(ccount==1)
			{
				ngap--;
				tmp=nmin*ngap;
			}
			*/
			caffe_set(1, tmp, MiddleMatS.mutable_cpu_data()+j*h+j);
		}
		//std::cout << ccount << " negative sigular elements in total." << std::endl;
		caffe_copy(dim,LeftMatU.cpu_data(),top_data_0+i*step_top_0);
		caffe_copy(dim,MiddleMatS.cpu_data(),top_data_1+i*step_top_1);
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
void EigLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}

	const int mnum = bottom[0]->num();
	const int h = bottom[0]->height();
	const int dim = h*h;

	const Dtype * top_diff_0=top[0]->cpu_diff();
	const Dtype * top_diff_1=top[1]->cpu_diff();
	const Dtype * top_data_0=top[0]->cpu_data();
	const Dtype * top_data_1=top[1]->cpu_data();
	const Dtype * bottom_data=bottom[0]->cpu_data();
	Dtype * bottom_diff=bottom[0]->mutable_cpu_diff();


	//Blob<Dtype> LeftMatU(1, 1, h, h);
	//Blob<Dtype> MiddleMatS(1, 1, h, h);
	//Blob<Dtype> LeftMatU_diff(1, 1, h, h);
	//Blob<Dtype> MiddleMatS_diff(1, 1, h, h);

	//Blob<Dtype> Identity(1, 1, h, h);
	//Blob<Dtype> Unit(1, 1, h, h);
	//Blob<Dtype> Sigma_i(1, 1, h, h);
	//Blob<Dtype> Sigma_j(1, 1, h, h);
	//Blob<Dtype> P(1, 1, h, h);

	//Blob<Dtype> temp_left(1, 1, h, h);
	//Blob<Dtype> temp_right(1, 1, h, h);

	Dtype *LeftMatU=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *MiddleMatS=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *LeftMatU_diff=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *MiddleMatS_diff=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));

	Dtype *Identity=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *Unit=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *Sigma_i=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *Sigma_j=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *P=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));

	Dtype *temp_left=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));
	Dtype *temp_right=reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * h * h));


	caffe_set(h*h,Dtype(1),Identity);
	caffe_set(h*h,Dtype(0),Unit);

	for(int i=0;i<h;i++){
		caffe_set(int(1.0),Dtype(1),Unit+i*h+i);
	}

	for(int i=0;i<mnum;i++){

		caffe_copy(h*h,top_data_0+i*dim,LeftMatU);
		caffe_copy(h*h,top_data_1+i*dim,MiddleMatS);
		caffe_copy(h*h,top_diff_0+i*dim,LeftMatU_diff);
		caffe_copy(h*h,top_diff_1+i*dim,MiddleMatS_diff);


		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								MiddleMatS,Identity,
								Dtype(0),Sigma_i);
		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								Identity,MiddleMatS,
								Dtype(0),Sigma_j);

		// for P ,the paper is T, so change the order of i and j
		caffe_sub<Dtype>(h*h,Sigma_j,Sigma_i,P);
		for(int j=0;j<h;j++){
			caffe_set(int(1),Dtype(1),P+j*h+j);
		}
		caffe_div<Dtype>(h*h,Identity,P,Sigma_j);
		for(int j=0;j<h;j++){
			caffe_set(int(1),Dtype(0),Sigma_j+j*h+j);
		}
		//calculate the left 
		caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								LeftMatU,LeftMatU_diff,
								Dtype(0),temp_left);

		caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								temp_left,Unit,
								Dtype(0),Sigma_i);

		caffe_add<Dtype>(h*h,temp_left,Sigma_i,temp_right);
		caffe_cpu_scale<Dtype>(h*h,Dtype(0.5),temp_right,temp_left);

		caffe_mul<Dtype>(h*h,Sigma_j,temp_left,temp_right);

		caffe_cpu_scale<Dtype>(h*h,Dtype(2.0),LeftMatU,Sigma_i);

		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								Sigma_i,temp_right,
								Dtype(0),Sigma_j);

		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasTrans,h,h,h,Dtype(1.0),
								Sigma_j,LeftMatU,
								Dtype(0),temp_left);

		//calculate the right
		caffe_mul<Dtype>(h*h,Unit,MiddleMatS_diff,temp_right);

		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,h,h,h,Dtype(1.0),
								LeftMatU,temp_right,
								Dtype(0),Sigma_i);

		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasTrans,h,h,h,Dtype(1.0),
								Sigma_i,LeftMatU,
								Dtype(0),temp_right);

		// sum the letf and right

		caffe_add<Dtype>(h*h,temp_left,temp_right,bottom_diff+i*dim);

	}
	
	free(LeftMatU);
	free(MiddleMatS);
	free(LeftMatU_diff);
	free(MiddleMatS_diff);

	free(Identity);
	free(Unit);
	free(Sigma_i);
	free(Sigma_j);
	free(P);
	
	free(temp_left);
	free(temp_right);

}


#ifdef CPU_ONLY
STUB_GPU(EigLayer);
#endif

INSTANTIATE_CLASS(EigLayer);
REGISTER_LAYER_CLASS(Eig);

}  // namespace caffe
