# OpenSubdiv

OpenSubdiv is a set of open source libraries that implement high performance subdivision surface (subdiv) evaluation on massively parallel CPU and GPU architectures. This codepath is optimized for drawing deforming subdivs with static topology at interactive framerates. The resulting limit surface matches Pixar's Renderman to numerical precision.

OpenSubdiv is covered by the Apache license, and is free to use for commercial or non-commercial use. This is the same code that Pixar uses internally for animated film production. Our intent is to encourage high performance accurate subdiv drawing by giving away the "good stuff".

Feel free to use it and let us know what you think.

For more details about OpenSubdiv, see [Pixar Graphics Technologies](http://graphics.pixar.com).


## Git Flow

We have adopted the git flow branching model. It is not necessary to use the git-flow extensions, though you may find them useful! But it will be helpful to read about the git flow branching model in order to understand the organization of branches and tags that you will find in the repository.

* [git-flow extensions](https://github.com/nvie/gitflow) 

## Quickstart

Basic instructions to get started with the code.

### Dependencies

Cmake will adapt the build based on which dependencies have been successfully discovered and will disable certain features and code examples accordingly.

Please refer to the documentation of each of the dependency packages for specific build and installation instructions.

Required:
* [cmake](http://www.cmake.org/cmake/resources/software.html)

Optional:
* [GLEW](http://sourceforge.net/projects/glew/) (Windows/Linux only)
* [CUDA](http://developer.nvidia.com/category/zone/cuda-zone)
* [OpenCL](http://www.khronos.org/opencl/)
* [GLFW](http://www.glfw.org/)
* [Ptex](https://github.com/wdas/ptex)
* [Zlib](http://www.zlib.net) (required for Ptex under Windows)
* [Maya SDK](http://www.autodesk.com/maya/) (sample code for Maya viewport 2.0 primitive)
* [DX11 SDK](http://www.microsoft.com/en-us/download/details.aspx?id=6812)
* [Docutils](http://docutils.sourceforge.net/)
* [Doxygen](file://www.doxygen.org/)

### Useful cmake options and environment variables

````
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
````

The paths to Maya, Ptex, GLFW, and GLEW can also be specified through the
following environment variables: `MAYA_LOCATION`, `PTEX_LOCATION`, `GLFW_LOCATION`,
and `GLEW_LOCATION`.


### Build instructions (Linux/OSX/Windows):

__Clone the repository:__

From the GitShell, Cygwin or the CLI :

````
git clone git://github.com/PixarAnimationStudios/OpenSubdiv.git
````

Alternatively, on Windows, GIT also provides a GUI to perform this operation.

__Generate Makefiles:__

Assuming that we want the binaries installed into a "build" directory at the root of the OpenSubdiv tree :
````
cd OpenSubdiv
mkdir build
cd build
````

Here is an example cmake configuration script for a full typical windows-based build that can be run in GitShell :

````
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
````

Alternatively, you can use the cmake GUI or run the commands from the CLI.

Note : the OSX builder in cmake for Xcode is -G "Xcode"

__Build the project:__

Windows : launch VC++ with the solution generated by cmake in your build directory.

OSX : run xcodebuild in your build directory

*Nix : run make in your build directory


## Standalone viewers

OpenSubdiv builds a number of standalone viewers that demonstrate various aspects of the software.

__Common Keyboard Shortcuts:__

````
Left mouse button drag   : orbit camera
Middle mouse button drag : pan camera
Right mouse button       : dolly camera
n, p                     : next/prev model
1, 2, 3, 4, 5, 6, 7      : specify adaptive isolation or uniform refinment level
+, -                     : increase / decrease tessellation 
w                        : switch display mode
q                        : quit
````

## Build instructions (iOS/Android)

OpenSubdiv may also be used for mobile app development.

Support for the CPU and GPU APIs used by OpenSubdiv is more limited on today's mobile operating systems.  For example, the most widely support graphics API is OpenGL ES 2.0 which doesn't yet provide the support for tessellation shaders needed to fully implement GPU accellerated Feature Adaptive Subdivision.

OpenSubdiv can still be used to compute uniform refinement of subdivision surfaces for display on these platforms, realizing all of the benefits of a consistent interpretation of subdivision schemes and tags.

The easiest way to get started using OpenSubdiv for mobile is to use CMake's support for cross-compiling:

* [CMake Cross Compiling](http://www.cmake.org/Wiki/CMake_Cross_Compiling)

### iOS

You will need a current version of Apple's Xcode and iOS SDK (tested with iOS 6.0.1 and Xcode 4.5.2):

* [Xcode](https://developer.apple.com/xcode/)

and a CMake toolchain for iOS:

* [iOS CMake](https://code.google.com/p/ios-cmake/)

You can then use CMake to configure and generate an Xcode project:

````
mkdir build-ios
cd build-ios
cmake -DCMAKE_TOOLCHAIN_FILE=[path to iOS.cmake] -GXcode ..

xcodebuild -target install -configuration Debug
````

You can open the resulting Xcode project directly, or include as a sub-project in the Xcode project for your app.

### Android NDK

You will need a current version of the Android NDK (tested with Android 4.2.1 and android-ndk-r8b):

* [Android NDK](http://developer.android.com/tools/sdk/ndk/index.html)

and a CMake toolchain for Android:

* [Android CMake](https://code.google.com/p/android-cmake/)

You can then use CMake to configure and build OpenSubdiv:

````
mkdir build-ndk
cd build-ndk
cmake -DCMAKE_TOOLCHAIN_FILE=[path to android.cmake] -DLIBRARY_OUTPUT_PATH_ROOT=`pwd`/modules/OpenSubdiv ..

make install
````

The resulting NDK module can be imported by other NDK modules by including it in your module search path:

````
export NDK_MODULE_PATH=[path to build-ndk/modules]
````


## Regression tests

OpenSubdiv builds a number of regression test executables for testing:

* hbr_regression: Regression testing matching HBR (low-level hierarchical boundary rep) to a pre-generated data set.
* far_regression: Matching FAR (feature-adaptive rep using tables) against HBR results.
* osd_regression: Matching full OSD subdivision against HBR results. Currently checks single threaded CPU kernel only.

