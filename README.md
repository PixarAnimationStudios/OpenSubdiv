# OpenSubdiv

OpenSubdiv is a set of open source libraries that implement high performance subdivision surface (subdiv) evaluation on massively parallel CPU and GPU architectures. This codepath is optimized for drawing deforming subdivs with static topology at interactive framerates. The resulting limit surface matches Pixar's Renderman to numerical precision.

OpenSubdiv is covered by the Apache license, and is free to use for commercial or non-commercial use. This is the same code that Pixar uses internally for animated film production. Our intent is to encourage high performance accurate subdiv drawing by giving away the "good stuff".

Feel free to use it and let us know what you think.

For more details about OpenSubdiv, see [Pixar Graphics Technologies](http://graphics.pixar.com).

 * master : [![Build Status](https://travis-ci.org/PixarAnimationStudios/OpenSubdiv.svg?branch=master)](https://travis-ci.org/PixarAnimationStudios/OpenSubdiv)

 * dev : [![Build Status](https://travis-ci.org/PixarAnimationStudios/OpenSubdiv.svg?branch=dev)](https://travis-ci.org/PixarAnimationStudios/OpenSubdiv)

## Documents
 * [User Documents] (http://graphics.pixar.com/opensubdiv/docs/intro.html)
 * [Doxygen API Documents] (http://graphics.pixar.com/opensubdiv/docs/doxy_html/index.html)
 * [Release Notes] (http://graphics.pixar.com/opensubdiv/docs/release_notes.html)

## Forum
 * [OpenSubdiv Google Groups] (https://groups.google.com/forum/embed/?place=forum/opensubdiv)

## Prerequisite
  For complete information, please refer OpenSubdiv documents:
  [Building with CMake] (http://graphics.pixar.com/opensubdiv/docs/cmake_build.html)

 * General requirements:

| Lib (URL)                             | Min Version | Note       |
| ------------------------------------- | ----------- | ---------- |
| CMake <br> http://www.cmake.org       | 2.8.6       | *Required* |

 * Osd optional requirements:

| Lib (URL)                                            | Min Version    | Note           |
| ---------------------------------------------------- | -------------- | -------------- |
| GLEW <br> http://glew.sourceforge.net                | 1.9.0          | GL backend (Win/Linux only) |
| CUDA <br> http://developer.nvidia.com/cuda-toolkit   | 4.0            | cuda backend   |
| TBB  <br> https://www.threadingbuildingblocks.org    | 4.0            | TBB backend    |
| OpenCL <br> http://www.khronos.org/opencl            | 1.1            | CL backend     |
| DX11 SDK <br> http://www.microsoft.com/download/details.aspx?id=6812| | DX backend     |

 * Examples/Documents optional requirements:

| Lib (URL)                                     | Min Version | Note                |
| --------------------------------------------- | ----------- | ------------------- |
| GLFW <br> http://www.glfw.org                 | 3.0.0       | GL examples         |
| Maya SDK <br> http://www.autodesk.com/maya    | 2013        | maya plugin example |
| Ptex <br> https://github.com/wdas/ptex        | 2.0         | ptex viewers        |
| Zlib <br> http://www.zlib.net                 |             | (required for Ptex under windows)|
| Docutils <br> http://docutils.sourceforge.net |             | documents           |
| Doxygen <br>http://www.doxygen.org            |             | documents           |


## Build example to run glViewer and other example programs with minimal dependency

### All platforms:

  * Install cmake, GLFW and GLEW (GLEW is not required on OSX)

   make sure GLEW and GLFW install directories configured as follows:

```
   ${GLEW_LOCATION}/include/GL/glew.h
   ${GLEW_LOCATION}/lib/libGLEW.a (linux)
   ${GLEW_LOCATION}/lib/glew32.lib (windows)

   ${GLFW_LOCATION}/include/glfw3.h
   ${GLFW_LOCATION}/lib/libglfw3.a (linux)
   ${GLFW_LOCATION}/lib/glfw3.lib (windows)
```

  * Clone OpenSubdiv repository, and create a build directory.
```
   git clone https://github.com/PixarAnimationStudios/OpenSubdiv
   mkdir build
   cd build
```

### Windows (Visual Studio)

  * run cmake:
```
"c:/Program Files (x86)/CMake/bin/cmake.exe" ^
    -G "Visual Studio 12 Win64" ^
    -D NO_MAYA=1 -D NO_PTEX=1 -D NO_DOC=1 ^
    -D NO_OMP=1 -D NO_TBB=1 -D NO_CUDA=1 -D NO_OPENCL=1 -D NO_CLEW=1 ^
    -D "GLEW_LOCATION=*YOUR GLEW INSTALL LOCATION*" ^
    -D "GLFW_LOCATION=*YOUR GLFW INSTALL LOCATION*" ^
    ..
```
  * Open OpenSubdiv.sln in VisualStudio and build.

### Linux

```
cmake -D NO_MAYA=1 -D NO_PTEX=1 -D NO_DOC=1 \
      -D NO_OMP=1 -D NO_TBB=1 -D NO_CUDA=1 -D NO_OPENCL=1 -D NO_CLEW=1 \
      -D GLEW_LOCATION="*YOUR GLEW INSTALL LOCATION*" \
      -D GLFW_LOCATION="*YOUR GLFW INSTALL LOCATION*" \
      ..
make
```

### OSX

```
cmake -D NO_MAYA=1 -D NO_PTEX=1 -D NO_DOC=1 \
      -D NO_OMP=1 -D NO_TBB=1 -D NO_CUDA=1 -D NO_OPENCL=1 -D NO_CLEW=1 \
      -D GLFW_LOCATION="*YOUR GLFW INSTALL LOCATION*" \
      ..
make
```

### Useful cmake options and environment variables

````
-DCMAKE_BUILD_TYPE=[Debug|Release]

-DCMAKE_INSTALL_PREFIX=[base path to install OpenSubdiv]
-DCMAKE_LIBDIR_BASE=[library directory basename (default: lib)]

-DCUDA_TOOLKIT_ROOT_DIR=[path to CUDA Toolkit]
-DPTEX_LOCATION=[path to Ptex]
-DGLEW_LOCATION=[path to GLEW]
-DGLFW_LOCATION=[path to GLFW]
-DMAYA_LOCATION=[path to Maya]

-DNO_LIB=1        // disable the opensubdiv libs build (caveat emptor)
-DNO_EXAMPLES=1   // disable examples build
-DNO_TUTORIALS=1  // disable tutorials build
-DNO_REGRESSION=1 // disable regression tests build
-DNO_MAYA=1       // disable Maya plugin build
-DNO_PTEX=1       // disable PTex support
-DNO_DOC=1        // disable documentation build
-DNO_OMP=1        // disable OpenMP
-DNO_TBB=1        // disable TBB
-DNO_CUDA=1       // disable CUDA
-DNO_OPENCL=1     // disable OpenCL
-DNO_OPENGL=1     // disable OpenGL
-DNO_CLEW=1       // disable CLEW wrapper library
````

