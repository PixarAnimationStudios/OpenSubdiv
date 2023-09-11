..
     Copyright 2013 Pixar

     Licensed under the Apache License, Version 2.0 (the "Apache License")
     with the following modification; you may not use this file except in
     compliance with the Apache License and the following modification to it:
     Section 6. Trademarks. is deleted and replaced with:

     6. Trademarks. This License does not grant permission to use the trade
        names, trademarks, service marks, or product names of the Licensor
        and its affiliates, except as required to comply with Section 4(c) of
        the License and to reproduce the content of the NOTICE file.

     You may obtain a copy of the Apache License at

         http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the Apache License with the above modification is
     distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
     KIND, either express or implied. See the Apache License for the specific
     language governing permissions and limitations under the Apache License.


Building with CMake
-------------------

.. contents::
   :local:
   :backlinks: none


----

Overview
========

Assuming that you have `cloned <getting_started.html>`__ the source repository
and selected an appropriate release branch, the following instructions will
walk you through the CMake and configuration and build process.

CMake is a cross-platform, open-source build system. CMake controls the compilation
process using platform independent configuration files in order to generate
Makefiles and workspaces that are native to the platform of choice.

The process involves the following steps:

    #. Locate & build the requisite dependencies
    #. Configure & run CMake to generate Makefiles / MSVC solution / XCode project
    #. Run the build from make / MSVC / XCode

----

Step 1: Dependencies
====================

CMake will adapt the build based on which dependencies have been successfully
discovered and will disable certain features and code examples accordingly.

Please refer to the documentation of each of the dependency packages for specific
build and installation instructions.

Required
________
    - `CMake <http://www.cmake.org/>`__ version 3.12

Optional
________

    - `Ptex <http://ptex.us/>`__ (support features for ptex textures and the
      ptexViewer example)
    - `Zlib <http://www.zlib.net/>`__ (required for Ptex under Windows)
    - `CUDA <http://www.nvidia.com/object/cuda_home_new.html>`__
    - `TBB <http://www.threadingbuildingblocks.org/>`__
    - `OpenCL <http://www.khronos.org/opencl/>`__
    - `DX11 SDK <http://www.microsoft.com/>`__
    - `GLFW <https://github.com/glfw/glfw>`__ (required for standalone examples
      and some regression tests)
    - `Docutils <http://docutils.sourceforge.net/>`__ (required for reST-based documentation)
    - `Python Pygments <http://pygments.org/>`__ (required for Docutils reST styling)
    - `Doxygen <http://www.doxygen.org/>`__

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

The following configuration arguments can be passed to the CMake command line.

.. code:: c++

   -DCMAKE_BUILD_TYPE=[Debug|Release]

   -DCMAKE_INSTALL_PREFIX=[base path to install OpenSubdiv (default: Current directory)]
   -DCMAKE_LIBDIR_BASE=[library directory basename (default: lib)]

   -DCMAKE_PREFIX_PATH=[semicolon-separated list of directories specifying installation prefixes to be searched by the find_package() command (default: empty list)]

   -DCUDA_TOOLKIT_ROOT_DIR=[path to CUDA]
   -DOSD_CUDA_NVCC_FLAGS=[CUDA options, e.g. --gpu-architecture]

   -DPTEX_LOCATION=[path to Ptex]
   -DGLFW_LOCATION=[path to GLFW]
   -DICC_LOCATION=[path to Intel's C++ Studio XE]

   -DNO_LIB=1        // disable the opensubdiv libs build (caveat emptor)
   -DNO_EXAMPLES=1   // disable examples build
   -DNO_TUTORIALS=1  // disable tutorials build
   -DNO_REGRESSION=1 // disable regression tests build
   -DNO_PTEX=1       // disable PTex support
   -DNO_DOC=1        // disable documentation build
   -DNO_OMP=1        // disable OpenMP
   -DNO_TBB=1        // disable TBB
   -DNO_CUDA=1       // disable CUDA
   -DNO_OPENCL=1     // disable OpenCL
   -DNO_OPENGL=1     // disable OpenGL
   -DNO_CLEW=1       // disable CLEW wrapper library

Environment Variables
_____________________

The paths to Ptex, GLFW, other dependencies can also be specified
through the following environment variables:

.. code:: c++

   PTEX_LOCATION, GLFW_LOCATION

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
        -G "Visual Studio 15 2017 Win64" \
        -D "GLFW_LOCATION:string=c:/Program Files/glfw-2.7.7.bin.WIN64" \
        -D "OPENCL_INCLUDE_DIRS:string=c:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 4.2/OpenCL/common/inc" \
        -D "_OPENCL_CPP_INCLUDE_DIRS:string=c:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 4.2/OpenCL/common/inc" \
        -D "OPENCL_LIBRARIES:string=c:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 4.2/OpenCL/common/lib/x64/OpenCL.lib" \
        -D "PTEX_LOCATION:string=c:/Users/opensubdiv/demo/src/ptex/x64" \
        ..

    # copy Ptex dependencies (Windows only)
    mkdir -p bin/{Debug,Release}
    \cp -f c:/Users/opensubdiv/demo/src/zlib-1.2.7/contrib/vstudio/vc10/x64/ZlibDllRelease/zlibwapi.dll bin/Debug/
    \cp -f c:/Users/opensubdiv/demo/src/zlib-1.2.7/contrib/vstudio/vc10/x64/ZlibDllRelease/zlibwapi.dll bin/Release/
    \cp -f c:/Users/opensubdiv/demo/src/ptex/x64/lib/Ptex.dll bin/Debug/
    \cp -f c:/Users/opensubdiv/demo/src/ptex/x64/lib/Ptex.dll bin/Release/

.. container:: impnotip

   **Important**

      Notice that the following scripts start by **recursively removing** the *../build/* and
      *../inst/* directories. Make sure you modify them to suit your build workflow.

Here is a similar script for \*Nix-based platforms:

.. code:: c++

    echo "*** Removing build"
    cd ..; rm -rf build/ inst/; mkdir build; cd build;
    echo "*** Running cmake"
    cmake -DPTEX_LOCATION=/home/opensubdiv/dev/opensource/ptex/install \
          -DGLFW_LOCATION=/home/opensubdiv/dev/opensource/glfw/build \
          -DDOXYGEN_EXECUTABLE=/home/opensubdiv/dev/opensource/doxygen/inst/bin/doxygen \
          -DCMAKE_INSTALL_PREFIX=../inst \
          -DCMAKE_BUILD_TYPE=Debug \
          ..

Here is a similar script for macOS:

.. code:: c++

    echo "*** Removing build"
    cd ..; rm -rf build/ inst/; mkdir build; cd build;
    echo "*** Running cmake"
    cmake -DOPENGL_INCLUDE_DIR=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk/System/Library/Frameworks/OpenGL.framework/Headers \
          -DGLFW_LOCATION=/Users/opensubdiv/dev/opensource/glfw/inst \
          -DNO_OMP=1 -DNO_REGRESSION=0 \
          -DCMAKE_INSTALL_PREFIX=../inst \
          -DCMAKE_BUILD_TYPE=Debug \
           .."

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


Using Clang
___________

CMake can also be overriden to use the `clang <http://clang.llvm.org/>`__ compilers by configuring the following options:

.. code:: c++

    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \


----

Step 3: Building
================

CMake provides a cross-platform command-line build:

.. code:: c++

    cmake --build . --target install --config Release

Alternatively, you can use native toolkits to launch the build. The steps differ for each OS:

    * *Windows* :
        launch VC++ with the solution generated by CMake in your build directory.

    * *macOS* :
        launch Xcode with the xcodeproj generated by CMake in your build directory

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


----

Compiling & Linking an OpenSubdiv Application
=============================================

Here are example commands for building an OpenSubdiv application on several architectures:

**Linux**
::

  g++ -I$OPENSUBDIV/include -c myapp.cpp
  g++ myapp.o -L$OPENSUBDIV/lib -losdGPU -losdCPU -o myapp

**macOS**
::

  g++ -I$OPENSUBDIV/include -c myapp.cpp
  g++ myapp.o -L$OPENSUBDIV/lib -losdGPU -losdCPU -o myapp
  install_name_tool -add_rpath $OPENSUBDIV/lib myapp

(On 64-bit OS-X: add ``-m64`` after each ``g++``.)

**Windows**
::

  cl /nologo /MT /TP /DWIN32 /I"%OPENSUBDIV%\include" -c myapp.cpp
  link /nologo /out:myapp.exe /LIBPATH:"%OPENSUBDIV%\lib" libosdGPU.lib libosdCPU.lib myapp.obj


.. container:: impnotip

    **Note:**

    HBR uses the offsetof macro on a templated struct, which appears to spurriously set off a
    warning in both gcc and Clang. It is recommended to turn the warning off with the
    *-Wno-invalid-offsetof* flag.
