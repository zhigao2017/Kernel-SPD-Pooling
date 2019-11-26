
#include <vector>
#include "stdio.h"
#include "caffe/layers/kernelrbf_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template<typename Dtype>
void KernelRbfLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    const KernelRbfParameter& kernelrbf_param = this->layer_param_.kernelrbf_param();
    beta_=kernelrbf_param.beta();
    // nothing should be done here
}

template<typename Dtype>
void KernelRbfLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // check the shape of two bottom compatible
    for (int i = 0; i < 2; ++i) {
        const int num_axes = bottom[i]->num_axes();
        CHECK_EQ(num_axes, 4) << "Bilinear layer only support 4 dim blobs.";
    }
    for (int axis = 0; axis < 4; ++axis) {
        // the number of channels could be different
        if (axis == 1) {
            continue;
        }
        CHECK_EQ(bottom[0]->shape(axis), bottom[1]->shape(axis))
                << "Two bottom blobs not compatible at axis " << axis << ".";
    }

    // then assign the shape of the top blob
    vector<int> top_shape = bottom[0]->shape();
    top_shape[1] = 1;
    top_shape[2] = bottom[0]->shape(1);
    top_shape[3] = bottom[1]->shape(1);
    top[0]->Reshape(top_shape);
}

template<typename Dtype>
void KernelRbfLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    // write by gaozhi-------------
    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* bottom_data_0 = bottom[0]->cpu_data();
    const Dtype* bottom_data_1 = bottom[1]->cpu_data();
    const int step_top = top[0]->count(1);
    const int step_bottom_0 = bottom[0]->count(1);
    const int step_bottom_1 = bottom[1]->count(1);
    const int hw_bottom_0 = bottom[0]->count(2);
    const int hw_bottom_1 = bottom[1]->count(2);

    const int channel_bottom_0=bottom[0]->shape(1);
    const int channel_bottom_1=bottom[1]->shape(1);

    Dtype p_beta=beta_;

    Blob<Dtype> b_m_fi2;
    Blob<Dtype> b_m_fj2;
    Blob<Dtype> b_m_fifj;
    Blob<Dtype> b_m_chw;
    Blob<Dtype> b_v_c_hw;

    b_m_fi2.Reshape(1,1,channel_bottom_0,channel_bottom_0);
    b_m_fj2.Reshape(1,1,channel_bottom_0,channel_bottom_0);
    b_m_fifj.Reshape(1,1,channel_bottom_0,channel_bottom_0);
    b_m_chw.Reshape(1,1,channel_bottom_0,hw_bottom_0);
    b_v_c_hw.Reshape(1,1,channel_bottom_0,hw_bottom_0);

    Dtype * m_fi2=b_m_fi2.mutable_cpu_data();
    Dtype * m_fj2=b_m_fj2.mutable_cpu_data();
    Dtype * m_fifj=b_m_fifj.mutable_cpu_data();
    Dtype * m_chw=b_m_chw.mutable_cpu_data();
    Dtype * v_c_hw=b_v_c_hw.mutable_cpu_data();

    for (int b = 0; b < bottom[0]->shape(0); ++b) {
        //fifi------------------
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channel_bottom_0,
                channel_bottom_1, hw_bottom_0, Dtype(1.0),
                bottom_data_0 + b * step_bottom_0,
                bottom_data_1 + b * step_bottom_1, Dtype(0.0),
                m_fifj);

        //fi2--------------------
        caffe_powx<Dtype>(bottom[0]->count(1),bottom_data_0 + b * step_bottom_0, float(2),m_chw);
        caffe_set(bottom[0]->count(1),Dtype(1.0),v_c_hw);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channel_bottom_0,
                channel_bottom_0, hw_bottom_0, Dtype(1.0),
                m_chw,
                v_c_hw, Dtype(0.0),
                m_fi2);

        //fj2--------------------
        caffe_powx<Dtype>(bottom[1]->count(1),bottom_data_1 + b * step_bottom_1, float(2),m_chw);
        caffe_set(bottom[1]->count(1),Dtype(1.0),v_c_hw);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channel_bottom_1,
                channel_bottom_1, hw_bottom_1, Dtype(1.0),
                v_c_hw,
                m_chw,Dtype(0.0),
                m_fj2);
        //compose
        caffe_axpy<Dtype>(channel_bottom_0*channel_bottom_1,Dtype(1.0),m_fi2,top_data+b*step_top);
        caffe_axpy<Dtype>(channel_bottom_0*channel_bottom_1,Dtype(1.0),m_fj2,top_data+b*step_top);
        caffe_axpy<Dtype>(channel_bottom_0*channel_bottom_1,Dtype(-2.0),m_fifj,top_data+b*step_top);

    }
    caffe_scal<Dtype>(int(top[0]->count(0)),p_beta,top_data);

    caffe_exp<Dtype>(int(top[0]->count(0)),top_data,top_data);

}

template<typename Dtype>
void KernelRbfLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
   
    //write by gaozhi
    if ((!propagate_down[0]) && (!propagate_down[1]))
        return;
    vector<bool> pd = propagate_down;
    if (bottom[0] == bottom[1])
        pd[0] = pd[1] = true;
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff[2] = { bottom[0]->mutable_cpu_diff(), bottom[1]
            ->mutable_cpu_diff() };
    const Dtype* bottom_data[2] =
            { bottom[0]->cpu_data(), bottom[1]->cpu_data() };

    const Dtype* top_data=top[0]->cpu_data();

    const int step_top = top[0]->count(1);
    const int step_bottom[2] = { bottom[0]->count(1), bottom[1]->count(1) };

    const int hw_bottom[2]={ bottom[0]->count(2),bottom[1]->count(2)};

    const int channel_bottom[2]={bottom[0]->shape(1),bottom[1]->shape(1)};
    
    Dtype p_beta=beta_;

    Blob<Dtype> b_data_diff;
    Blob<Dtype> b_m_c_hw;
    Blob<Dtype> b_f_x;
    Blob<Dtype> b_f_y;

    b_data_diff.Reshape(1,1,channel_bottom[1],channel_bottom[0]);
    b_m_c_hw.Reshape(1,1,channel_bottom[0],hw_bottom[0]);
    b_f_x.Reshape(1,1,channel_bottom[0],hw_bottom[0]);
    b_f_y.Reshape(1,1,channel_bottom[0],hw_bottom[0]);

    Dtype * data_diff=b_data_diff.mutable_cpu_data();
    Dtype * m_c_hw=b_m_c_hw.mutable_cpu_data();
    Dtype * f_x=b_f_x.mutable_cpu_data();
    Dtype * f_y=b_f_y.mutable_cpu_data();
 
    for (int b = 0; b < bottom[0]->shape(0); ++b) {
        caffe_mul<Dtype>(channel_bottom[0]*channel_bottom[1],top_data+b*step_top,top_diff+b*step_top,data_diff);
        caffe_scal<Dtype>(channel_bottom[0]*channel_bottom[1],2*p_beta,data_diff);
        caffe_set(bottom[1]->count(1),Dtype(1.0),m_c_hw);
        //dx,fx
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channel_bottom[0],
                hw_bottom[0], channel_bottom[0], Dtype(1.0),
                data_diff,
                m_c_hw,Dtype(0.0),
                f_x);
        caffe_mul<Dtype>(bottom[0]->count(1),f_x,bottom_data[0]+b*step_bottom[0],f_x);
        //dx,fy
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channel_bottom[1],
                hw_bottom[0], channel_bottom[1], Dtype(1.0),
                data_diff,
                bottom_data[1]+b*step_bottom[1],Dtype(0.0),
                f_y);
        //dx,compose
        caffe_sub<Dtype>(bottom[0]->count(1),f_x,f_y,bottom_diff[0]+b*step_bottom[0]);
        //------------------------------------
        //dy,fy
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channel_bottom[1],
                hw_bottom[0],channel_bottom[1], Dtype(1.0),
                data_diff,
                m_c_hw,Dtype(0.0),
                f_y);
        caffe_mul<Dtype>(bottom[1]->count(1),f_y,bottom_data[1]+b*step_bottom[1],f_y);
        //dy,fx
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channel_bottom[0],
                hw_bottom[1], channel_bottom[0], Dtype(1.0),
                data_diff,
                bottom_data[0]+b*step_bottom[0],Dtype(0.0),
                f_x);
        //dy,compose
        caffe_sub<Dtype>(bottom[1]->count(1),f_y,f_x,bottom_diff[1]+b*step_bottom[1]);

    }
}

#ifdef CPU_ONLY
STUB_GPU(KernelRbfLayer);
#endif

INSTANTIATE_CLASS(KernelRbfLayer);
REGISTER_LAYER_CLASS(KernelRbf);

}  // namespace caffe
