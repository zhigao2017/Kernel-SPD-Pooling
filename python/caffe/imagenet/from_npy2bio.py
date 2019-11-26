import numpy as np
import caffe
import sys
meanProtoPath = 'ilsvrc_2012_mean.binaryproto'
meanNpyPath='ilsvrc_2012_mean.npy'
blob = caffe.proto.caffe_pb2.BlobProto()
with open(meanNpyPath,'rb') as f:
    mean = np.load(f)
blob.channels=1
blob.height=mean.shape[0]
blob.width = mean.shape[1]
blob.data.extend(mean.astype(float).flat)
binaryprotoFile = open(meanProtoPath,'wb')
binaryprotoFile.write(blob.SerializeToString())
binaryprotoFile.close()