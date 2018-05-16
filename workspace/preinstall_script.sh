#!/bin/bash
# May need to uncomment and update to find current packages
apt-get update

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.4-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
wget https://developer.nvidia.com/compute/machine-learning/tensorrt/3.0/ga/nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64-deb
dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
dpkg -i libnccl2_2.1.4-1+cuda9.0_amd64.deb
dpkg -i libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
dpkg -i nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64.deb
apt-get update
apt-get -y install cuda=9.0.176-1
apt-get -y install libcudnn7-dev
apt-get -y install libnccl-dev

# Required for demo script! #
pip install scikit-video
pip install tensorflow-gpu==1.7.0

# Add your desired packages for each workspace initialization
#          Add here!          #
wget https://storage.googleapis.com/fkhan-udacity/fcn8.pb