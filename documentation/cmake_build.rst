..
     Copyright 2013 Pixar

     Licensed under the terms set forth in the LICENSE.txt file available at
     https://opensubdiv.org/license.


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
walk you through the CMake configuration and build process.

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
    - `CMake <https://www.cmake.org/>`__ version 3.14

Optional OpenSubdiv::Osd Dependencies
_____________________________________

    - `OpenGL <https://www.opengl.org>`__
    - `Metal <https://developer.apple.com/metal>`__
    - `CUDA <https://developer.nvidia.com/cuda-toolkit>`__
    - `TBB <https://github.com/uxlfoundation/oneTBB>`__
    - `OpenCL <https://www.khronos.org/opencl/>`__
    - `DX11 SDK <https://www.microsoft.com/en-us/download/details.aspx?id=6812>`__

Optional Interactive Example Dependencies
_________________________________________

    - `GLFW <https://www.glfw.org>`__ (for OpenGL example viewers and some regression tests)
    - `Ptex <https://ptex.us/>`__ (for Ptex example viewers)
    - `Zlib <https://www.zlib.net/>`__ (for Ptex example viewers)

Optional Documentation Dependencies
___________________________________

    - `Doxygen <http://www.doxygen.org/>`__ (for C++ API documentation)
    - `Docutils <https://pypi.org/project/docutils>`__ (for reStructuredText documentation)
    - `Python Pygments <https://pypi.org/project/Pygments>`__ (for reStructuredText documentation styling)

----

Step 2: Configuring CMake
=========================

One way to configure CMake is to use the `CMake GUI <https://cmake.org/cmake/help/latest/manual/cmake-gui.1.html>`__.
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
   -DCMAKE_SYSTEM_NAME=[target system name for cross-compilation builds, e.g. iOS]

   -DCMAKE_PREFIX_PATH=[semicolon-separated list of directories specifying installation prefixes to be searched by the find_package() command (default: empty list)]

   -DCUDA_SDK_ROOT_DIR=[path to CUDA]
   -DCUDA_TOOLKIT_ROOT_DIR=[path to CUDA]
   -DOSD_CUDA_NVCC_FLAGS=[CUDA options, e.g. --gpu-architecture]

   -DGLFW_LOCATION=[path to GLFW for OpenGL example viewers]
   -DPTEX_LOCATION=[path to Ptex for Ptex example viewers]
   -DICC_LOCATION=[path to Intel's C++ Studio XE]

   -DNO_LIB=1        // disable the opensubdiv libs build (caveat emptor)
   -DNO_EXAMPLES=1   // disable examples build
   -DNO_TUTORIALS=1  // disable tutorials build
   -DNO_REGRESSION=1 // disable regression tests build
   -DNO_PTEX=1       // disable Ptex examples
   -DNO_DOC=1        // disable documentation build
   -DNO_OMP=1        // disable OpenMP
   -DNO_TBB=1        // disable TBB
   -DNO_CUDA=1       // disable CUDA
   -DNO_OPENCL=1     // disable OpenCL
   -DNO_CLEW=1       // disable OpenCL loader library
   -DNO_OPENGL=1     // disable OpenGL
   -DNO_METAL=1      // disable Metal

   -DOSD_PATCH_SHADER_SOURCE_GLSL=1  // GLSL Patch Shader Source
   -DOSD_PATCH_SHADER_SOURCE_HLSL=1  // HLSL Patch Shader Source
   -DOSD_PATCH_SHADER_SOURCE_MSL=1   // MSL Patch Shader Source

Environment Variables
_____________________

The paths to GLFW, Ptex, Zlib, and other dependencies can also be specified
through the following environment variables:

.. code:: c++

   GLFW_LOCATION, PTEX_LOCATION, ZLIB_ROOT

Automated Script
________________

The GUI solution will probably become a burden for active developpers who tend to
re-run the configuration step fairly often. A scripted solution can save a lot of
time. Here is a typical workflow:

.. code:: c++

    git clone https://github.com/PixarAnimationStudios/OpenSubdiv.git <folder>
    cd <folder>
    source cmake_setup

Where *cmake_setup* is a configuration script.

.. container:: impnotip

   **Important**

      Notice that the following scripts start by **recursively removing** the *buildDir* and
      *instDir* directories. Make sure you modify them to suit your build workflow.

Here is an example CMake configuration script for a full typical windows-based
build that can be run in GitShell :

.. code:: c++

    echo "*** Removing build"
    rm -rf buildDir instDir
    echo "*** Running cmake"
    cmake -B buildDir \
          -D CMAKE_INSTALL_PREFIX=instDir \
          -G "Visual Studio 16 2019" -A x64 \
          -D "GLFW_LOCATION=C:\path\to\glfw" \
          -S .

Here is a similar script for \*Nix-based platforms:

.. code:: c++

    echo "*** Removing build"
    rm -rf buildDir instDir
    echo "*** Running cmake"
    cmake -B buildDir \
          -D CMAKE_INSTALL_PREFIX=instDir \
          -D "GLFW_LOCATION=/path/to/glfw" \
          -S .

Here is a similar script for macOS:

.. code:: c++

    echo "*** Removing build"
    rm -rf buildDir instDir
    echo "*** Running cmake"
    cmake -B buildDir \
          -D CMAKE_INSTALL_PREFIX=instDir \
          -G Xcode \
          -D "GLFW_LOCATION=/path/to/glfw" \
          -S .

Here is a similar script for iOS:

Use CMAKE_SYSTEM_NAME to have CMake use the appropriate cross-compilation toolchain when building for iOS.

.. code:: c++

    echo "*** Removing build"
    rm -rf buildDir instDir
    echo "*** Running cmake"
    SDKROOT=$(xcrun --sdk iphoneos --show-sdk-path)
    cmake -B buildDir \
          -D CMAKE_INSTALL_PREFIX=instDir \
          -G Xcode \
          -D CMAKE_SYSTEM_NAME=iOS \
          -D NO_PTEX=1 -D NO_DOC=1 \
          -D NO_OMP=1 -D NO_TBB=1 -D NO_CUDA=1 -D NO_OPENCL=1 -D NO_CLEW=1 \
          -D NO_TUTORIALS=1 -D NO_EXAMPLES=1 -D NO_REGRESSION=1 -D NO_OPENGL=1 \
          -S .

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

    cmake --build buildDir --target install --config Release

Alternatively, you can use native toolkits to launch the build. The steps differ for each OS:

    * *Windows* :
        launch VIsual Studio IDE with the solution generated by CMake in your build directory.

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
