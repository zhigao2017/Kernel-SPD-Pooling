#!/bin/bash

# first fine all layers

./../../../caffe/build/tools/caffe train \
	-model "minc_conv_all.prototxt" \
	-solver "minc_conv_all.solver" \
	-weights "/home/zealot/gaozhi/model/VGG_ILSVRC_16_layers.caffemodel" \
	-gpu 0
	