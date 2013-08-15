..  
       Copyright 2013 Pixar

       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License
       and the following modification to it: Section 6 Trademarks.
       deleted and replaced with:

       6. Trademarks. This License does not grant permission to use the
       trade names, trademarks, service marks, or product names of the
       Licensor and its affiliates, except as required for reproducing
       the content of the NOTICE file.

       You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

       Unless required by applicable law or agreed to in writing,
       software distributed under the License is distributed on an
       "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
       either express or implied.  See the License for the specific
       language governing permissions and limitations under the
       License.
  

Building with Cmake
-------------------

.. contents::
   :local:
   :backlinks: none


Information on how to build OpenSubdiv

----

Overview
========

Assuming that you have `cloned <getting_started.html>`__ the source repository 
and selected an appropriate release branch, the following instructions will
walk you through the Cmake and configuration and build process.

Cmake is a cross-platform, open-source build system. Cmake controls the compilation
process using platform independent configuration files in order to generate 
makefiles and workspaces that are native to the platform of choice.

The process involves the following steps:
    1. Locate & build the requisite dependencies
    2. Configure & run CMake to generate Makefiles / MSVC solution / XCode project
    3. Run the build from make / MSVC / XCode

----

Step 1: Dependencies
====================

Cmake will adapt the build based on which dependencies have been successfully 
discovered and will disable certain features and code examples accordingly.

Please refer to the documentation of each of the dependency packages for specific 
build and installation instructions.

Required
________
    - `cmake <http://www.cmake.org/>`__ version 2.8

Optional
________

    - `Ptex <http://ptex.us/>`__ (support features for ptex textures and the
      ptexViewer example)
    - `Zlib <http://www.zlib.net/>`__ (required for Ptex under Windows)
    - `GLEW <http://glew.sourceforge.net/>`__ (Windows/Linux only)
    - `CUDA <http://www.nvidia.com/object/cuda_home_new.html>`__
    - `TBB <http://www.threadingbuildingblocks.org/>`__
    - `OpenCL <http://www.khronos.org/opencl/>`__
    - `DX11 SDK <http://www.microsoft.com/>`__
    - `GLFW <https://github.com/glfw/glfw>`__ (required for standalone examples
      and some regression tests)
    - `Docutils <http://docutils.sourceforge.net/>`__ (required for reST-based documentation)
    - `Python Pygments <http://www.pygments.org/>`__ (required for Docutils reST styling)
    - `Doxygen <www.doxygen.org/>`__

----

Step 2: Configuring CMake
=========================

One way to configure CMake is to use the `CMake GUI <http://www.cmake.org/cmake/help/runningcmake.html>`__.
In many cases CMake can fall back on default standard paths in order to find the
packages that OpenSubdiv depends on. For non-standard installations however, a
complete set of override variables is available. The following sub-section lists
some of these variables. For more specific details, please consult the source of
the custom CMake modules in the OpenSubdiv/cmake/ folder.

Useful Build Options
____________________

The following configuration arguments can be passed to the cmake command line.

.. code:: c++

   -DCMAKE_BUILD_TYPE=[Debug|Release]

   -DCMAKE_INSTALL_PREFIX=[base path to install OpenSubdiv (default: Current directory)]
   -DCMAKE_LIBDIR_BASE=[library directory basename (default: lib)]
   
   -DCUDA_TOOLKIT_ROOT_DIR=[path to CUDA]
   -DPTEX_LOCATION=[path to Ptex]
   -DGLEW_LOCATION=[path to GLEW]
   -DGLFW_LOCATION=[path to GLFW]
   -DMAYA_LOCATION=[path to Maya]
   -DTBB_LOCATION=[path to Intel's TBB]
   -DICC_LOCATION=[path to Intel's C++ Studio XE]
   
   -DNO_LIB=1        // disable the opensubdiv libs build (caveat emptor)
   -DNO_EXAMPLES=1   // disable examples build
   -DNO_REGRESSION=1 // disable regression tests build
   -DNO_PYTHON=1     // disable Python SWIG build
   -DNO_DOC=1        // disable documentation build
   -DNO_OMP=1        // disable OpenMP
   -DNO_TBB=1        // disable TBB
   -DNO_CUDA=1       // disable CUDA
   -DNO_GCD=1        // disable GrandCentralDispatch on OSX

Environment Variables
_____________________

The paths to Maya, Ptex, GLFW, and GLEW can also be specified through the 
following environment variables: 

.. code:: c++

   MAYA_LOCATION
   PTEX_LOCATION
   GLFW_LOCATION
   GLEW_LOCATION.
   
Automated Script
________________

The GUI solution will probably become a burden for active developpers who tend to
re-run the configuration step fairly often. A scripted solution can save a lot of
time. Here is a typical workflow:

.. code:: c++

    git clone https://github.com/PixarAnimationStudios/OpenSubdiv.git <folder>
    cd <folder>
    mkdir build
    cd build
    source ../../cmake_setup


Where *cmake_setup* is a configuration script.

Here is an example CMake configuration script for a full typical windows-based 
build that can be run in GitShell :

.. code:: c++

    #/bin/tcsh

    # Replace the ".." with a full path to the root of the OpenSubdiv source tree if necessary
    "c:/Program Files (x86)/CMake 2.8/bin/cmake.exe" \
        -G "Visual Studio 10 Win64" \
        -D "GLEW_LOCATION:string=c:/Program Files/glew-1.9.0" \
        -D "GLFW_LOCATION:string=c:/Program Files/glfw-2.7.7.bin.WIN64" \
        -D "OPENCL_INCLUDE_DIRS:string=c:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 4.2/OpenCL/common/inc" \
        -D "_OPENCL_CPP_INCLUDE_DIRS:string=c:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 4.2/OpenCL/common/inc" \
        -D "OPENCL_LIBRARIES:string=c:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 4.2/OpenCL/common/lib/x64/OpenCL.lib" \
        -D "MAYA_LOCATION:string=c:/Program Files/Autodesk/Maya2013.5" \
        -D "PTEX_LOCATION:string=c:/Users/opensubdiv/demo/src/ptex/x64" \
        ..

    # copy Ptex dependencies (Windows only)
    mkdir -p bin/{Debug,Release}
    \cp -f c:/Users/opensubdiv/demo/src/zlib-1.2.7/contrib/vstudio/vc10/x64/ZlibDllRelease/zlibwapi.dll bin/Debug/
    \cp -f c:/Users/opensubdiv/demo/src/zlib-1.2.7/contrib/vstudio/vc10/x64/ZlibDllRelease/zlibwapi.dll bin/Release/
    \cp -f c:/Users/opensubdiv/demo/src/ptex/x64/lib/Ptex.dll bin/Debug/
    \cp -f c:/Users/opensubdiv/demo/src/ptex/x64/lib/Ptex.dll bin/Release/
    

Here is a similar script for \*Nix-based platforms:

.. code:: c++

    echo "*** Removing build"
    cd ..; rm -rf build/ inst/; mkdir build; cd build;
    echo "*** Running cmake"
    cmake -DPTEX_LOCATION=/home/opensubdiv/dev/opensource/ptex/install \
          -DGLEW_LOCATION=/home/opensubdiv/dev/opensource/glew/glew-1.9.0 \
          -DGLFW_LOCATION=/home/opensubdiv/dev/opensource/glfw/build \
          -DDOXYGEN_EXECUTABLE=/home/opensubdiv/dev/opensource/doxygen/inst/bin/doxygen \
          -DCMAKE_INSTALL_PREFIX=../inst \
          -DCMAKE_BUILD_TYPE=Debug \
          ..

.. container:: impnotip

   * **Important**

      Notice that this script starts by **recursively removing** the *../build/* and 
      *../inst/* directories. Make sure you modify this script to suit your build
      workflow.

Using Intel's C++ Studio XE
___________________________

OpenSubdiv can be also be built with `Intel's C++ compiler <http://software.intel.com/en-us/intel-compilers>`__ 
(icc). The default compiler can be overriden in CMake with the following configuration options:

.. code:: c++

    -DCMAKE_CXX_COMPILER=[path to icc executable]
    -DCMAKE_C_COMPILER=[path to icc executable]

The installation location of the C++ Studio XE can be overriden with:

.. code:: c++

    -DICC_LOCATION=[path to Intel's C++ Studio XE]


----

Step 3: Building
================

The steps differ for different OS'es:

    * *Windows* : 
        launch VC++ with the solution generated by cmake in your build directory.

    * *OSX* : 
        run xcodebuild in your build directory

    * *\*Nix* : 
        | run *make* in your build directory 
        | - use the *clean* target to remove previous build results
        | - use *VERBOSE=1* for verbose build output
    
----

Build Targets
_____________

Makefile-based builds allow the use of named target. Here are some of the more
useful target names:
    
   *osd_\<static\|dynamic\>_\<CPU\|GPU\>*
      | The core components of the OpenSubdiv libraries
      |

   *\<example_name\>*
      | Builds specific code examples by name (glViewer, ptexViewer...)
      |
      
   *doc*
      | Builds ReST and doxygen documentation
      |

   *doc_html*
      | Builds ReST documentation
      |

   *doc_doxy*
      | Builds Doxygen documentation
      |

