..  
       Copyright (C) Pixar. All rights reserved.
  
       This license governs use of the accompanying software. If you
       use the software, you accept this license. If you do not accept
       the license, do not use the software.
  
       1. Definitions
       The terms "reproduce," "reproduction," "derivative works," and
       "distribution" have the same meaning here as under U.S.
       copyright law.  A "contribution" is the original software, or
       any additions or changes to the software.
       A "contributor" is any person or entity that distributes its
       contribution under this license.
       "Licensed patents" are a contributor's patent claims that read
       directly on its contribution.
  
       2. Grant of Rights
       (A) Copyright Grant- Subject to the terms of this license,
       including the license conditions and limitations in section 3,
       each contributor grants you a non-exclusive, worldwide,
       royalty-free copyright license to reproduce its contribution,
       prepare derivative works of its contribution, and distribute
       its contribution or any derivative works that you create.
       (B) Patent Grant- Subject to the terms of this license,
       including the license conditions and limitations in section 3,
       each contributor grants you a non-exclusive, worldwide,
       royalty-free license under its licensed patents to make, have
       made, use, sell, offer for sale, import, and/or otherwise
       dispose of its contribution in the software or derivative works
       of the contribution in the software.
  
       3. Conditions and Limitations
       (A) No Trademark License- This license does not grant you
       rights to use any contributor's name, logo, or trademarks.
       (B) If you bring a patent claim against any contributor over
       patents that you claim are infringed by the software, your
       patent license from such contributor to the software ends
       automatically.
       (C) If you distribute any portion of the software, you must
       retain all copyright, patent, trademark, and attribution
       notices that are present in the software.
       (D) If you distribute any portion of the software in source
       code form, you may do so only under this license by including a
       complete copy of this license with your distribution. If you
       distribute any portion of the software in compiled or object
       code form, you may only do so under a license that complies
       with this license.
       (E) The software is licensed "as-is." You bear the risk of
       using it. The contributors give no express warranties,
       guarantees or conditions. You may have additional consumer
       rights under your local laws which this license cannot change.
       To the extent permitted under your local laws, the contributors
       exclude the implied warranties of merchantability, fitness for
       a particular purpose and non-infringement.
  

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

    - `GLEW <http://glew.sourceforge.net/>`__ (Windows/Linux only)
    - `CUDA <http://www.nvidia.com/object/cuda_home_new.html>`__
    - `OpenCL <http://www.khronos.org/opencl/>`__
    - `GLFW <https://github.com/glfw/glfw>`__ (required for standalone examples
      and some regression tests)
    - `Ptex <http://ptex.us/>`__ (support features for ptex textures and the
      ptexViewer example)
    - `Zlib <http://www.zlib.net/>`__ (required for Ptex under Windows)
    - `DX11 SDK <http://www.microsoft.com/>`__
    - `Docutils <http://docutils.sourceforge.net/>`__

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
   
   -DNO_LIB=1        // disable the opensubdiv libs build (caveat emptor)
   -DNO_EXAMPLES=1   // disable examples build
   -DNO_REGRESSION=1 // disable regression tests build
   -DNO_PYTHON=1     // disable Python SWIG build
   -DNO_DOC=1        // disable documentation build
   -DNO_OMP=1        // disable OpenMP
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

