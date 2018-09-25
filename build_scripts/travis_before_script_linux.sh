#
#   Copyright 2017 Pixar
#
#   Licensed under the Apache License, Version 2.0 (the "Apache License")
#   with the following modification; you may not use this file except in
#   compliance with the Apache License and the following modification to it:
#   Section 6. Trademarks. is deleted and replaced with:
#
#   6. Trademarks. This License does not grant permission to use the trade
#      names, trademarks, service marks, or product names of the Licensor
#      and its affiliates, except as required to comply with Section 4(c) of
#      the License and to reproduce the content of the NOTICE file.
#
#   You may obtain a copy of the Apache License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the Apache License with the above modification is
#   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied. See the Apache License for the specific
#   language governing permissions and limitations under the Apache License.
#


# opensubdiv needs 2.8.8 or later.
cmake --version

###############################################################################

sudo apt-get update -qq

# Install X11 dev libraries
sudo apt-get install libxrandr-dev
sudo apt-get install libxcursor-dev
sudo apt-get install libxinerama-dev
sudo apt-get install libxi-dev

# install glut and xxf86vm (for GL libs)
sudo apt-get install freeglut3
sudo apt-get install freeglut3-dev
sudo apt-get install libxxf86vm1
sudo apt-get install libxxf86vm-dev

# install GLEW
sudo apt-get install libglew1.10
sudo apt-get install libglew-dev


###############################################################################
# Upgrade to get a version of Mesa that supports OGL 4
sudo add-apt-repository ppa:ubuntu-x-swat/updates -y
sudo apt-get update -qq
sudo apt-get dist-upgrade


###############################################################################
# Build and install glfw
mkdir glfw && pushd glfw
git clone https://github.com/glfw/glfw
mkdir build && cd build
cmake ../glfw
make
sudo make install
popd



###############################################################################
# Start an X Virtual Framebuffer so that we can do some basic imaging tests.
export DISPLAY=:99.0
sh -e /etc/init.d/xvfb start

###############################################################################
# Install TBB 4.3 update 1
wget https://www.threadingbuildingblocks.org/sites/default/files/software_releases/linux/tbb43_20141023oss_lin.tgz -O /tmp/tbb.tgz
tar -xvzf /tmp/tbb.tgz -C $HOME

###############################################################################
# Install PTex
wget https://github.com/wdas/ptex/archive/v2.0.30.tar.gz -O /tmp/ptex.tgz
tar -xvzf /tmp/ptex.tgz -C $HOME
pushd $HOME/ptex-2.0.30/src
make
mkdir $HOME/ptex
mv $HOME/ptex-2.0.30/install/* $HOME/ptex
popd
