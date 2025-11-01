# OpenSubdiv

OpenSubdiv is a set of open source libraries that implement high performance subdivision surface (subdiv) evaluation on massively parallel CPU and GPU architectures. This codepath is optimized for drawing deforming subdivs with static topology at interactive framerates. The resulting limit surface matches Pixar's Renderman to numerical precision.

For more details, please visit the web site [opensubdiv.org](https://opensubdiv.org).

|         |   Linux   |  Windows  |   macOS   |
|:-------:|:---------:|:---------:|:---------:|
|   dev   | [![Build Status](https://dev.azure.com/PixarAnimationStudios/OpenSubdiv/_apis/build/status/PixarAnimationStudios.OpenSubdiv?branchName=dev&amp;jobName=Linux)](https://dev.azure.com/PixarAnimationStudios/OpenSubdiv/_build/latest?definitionId=2&branchName=dev) | [![Build Status](https://dev.azure.com/PixarAnimationStudios/OpenSubdiv/_apis/build/status/PixarAnimationStudios.OpenSubdiv?branchName=dev&amp;jobName=Windows)](https://dev.azure.com/PixarAnimationStudios/OpenSubdiv/_build/latest?definitionId=2&branchName=dev) | [![Build Status](https://dev.azure.com/PixarAnimationStudios/OpenSubdiv/_apis/build/status/PixarAnimationStudios.OpenSubdiv?branchName=dev&amp;jobName=macOS)](https://dev.azure.com/PixarAnimationStudios/OpenSubdiv/_build/latest?definitionId=2&branchName=dev) |
|  release | [![Build Status](https://dev.azure.com/PixarAnimationStudios/OpenSubdiv/_apis/build/status/PixarAnimationStudios.OpenSubdiv?branchName=release&amp;jobName=Linux)](https://dev.azure.com/PixarAnimationStudios/OpenSubdiv/_build/latest?definitionId=2&branchName=release) | [![Build Status](https://dev.azure.com/PixarAnimationStudios/OpenSubdiv/_apis/build/status/PixarAnimationStudios.OpenSubdiv?branchName=release&amp;jobName=Windows)](https://dev.azure.com/PixarAnimationStudios/OpenSubdiv/_build/latest?definitionId=2&branchName=release) | [![Build Status](https://dev.azure.com/PixarAnimationStudios/OpenSubdiv/_apis/build/status/PixarAnimationStudios.OpenSubdiv?branchName=release&amp;jobName=macOS)](https://dev.azure.com/PixarAnimationStudios/OpenSubdiv/_build/latest?definitionId=2&branchName=release) |

## Documents
 * [User Documents](https://opensubdiv.org/docs/intro.html)
 * [Doxygen API Documents](https://opensubdiv.org/docs/doxy_html/index.html)
 * [Release Notes](https://opensubdiv.org/docs/release_notes.html)

## Forum
 * [OpenSubdiv Google Groups](https://groups.google.com/forum/embed/?place=forum/opensubdiv)

## Prerequisites and Building

The OpenSubdiv core libraries are implemented in C++ with no dependencies other than the C++ standard library.

The OpenSubdiv::Osd library contains additional conditionally compiled components which use specific external CPU and GPU APIs for evaluation and display and there are also optional interactive examples. Some of these optional aspects are enabled by default but can be disabled while configuring the OpenSubdiv build.

For complete information, please see:
[Building with CMake](https://opensubdiv.org/docs/cmake_build.html)

## Versions

These are the versions of external dependencies used to test the current release of OpenSubdiv. Generally, OpenSubdiv will also work with earlier and later versions of these external dependencies, but this represents versions known to work.

| Core Dependencies                                                            | Version     | Note                   |
| ---------------------------------------------------------------------------- | ----------- | ---------------------- |
| [CMake](https://www.cmake.org)                                               | 3.14        | *Required*             |

| Optional OpenSubdiv::Osd Dependencies                                        | Version     | Note                   |
| ---------------------------------------------------------------------------- | ----------- | ---------------------- |
| [OpenGL](https://www.opengl.org)                                             | 4.1+        | OpenGL                 |
| [Metal](https://developer.apple.com/metal)                                   | 3+          | Metal                  |
| [CUDA](https://developer.nvidia.com/cuda-toolkit)                            | 12.6        | CUDA                   |
| [TBB](https://github.com/uxlfoundation/oneTBB)                               | 2021.12.0   | oneTBB                 |
| [OpenCL](https://www.khronos.org/opencl)                                     | 1.1         | OpenCL                 |
| [DirectX11](https://www.microsoft.com/en-us/download/details.aspx?id=6812)   | 11+         | DirectX11              |

| Optional Interactive Example Dependencies                                    | Version     | Note                   |
| ---------------------------------------------------------------------------- | ----------- | ---------------------- |
| [GLFW](https://www.glfw.org)                                                 | 3.3.3       | OpenGL example viewers |
| [Ptex](https://github.com/wdas/ptex)                                         | 2.4.2       | Ptex example viewers   |
| [Zlib](https://www.zlib.net)                                                 | 1.2.13      | Ptex example viewers   |

| Optional Documentation Dependencies                                          | Version     | Note                   |
| ---------------------------------------------------------------------------- |------------ |----------------------- |
| [Doxygen](https://www.doxygen.nl)                                            | 1.14.0      | C++ API Documentation  |
| [Docutils](https://pypi.org/project/docutils)                                | 0.21.2      | general documentation  |
| [Pygments](https://pypi.org/project/Pygments)                                | 2.19        | general documentation  |

## Building with minimal dependencies to run glViewer and other example programs

### All platforms:

  * Install cmake and GLFW

   make sure GLFW install directories are configured as follows:

```
   ${GLFW_LOCATION}/include/GLFW/glfw3.h
   ${GLFW_LOCATION}/lib/libglfw3.a (linux)
   ${GLFW_LOCATION}/lib/glfw3.lib (windows)
```

  * Clone OpenSubdiv repository
```
   git clone https://github.com/PixarAnimationStudios/OpenSubdiv
   cd OpenSubdiv
```

### Windows (Visual Studio)

```
cmake -B buildDir ^
      -D CMAKE_INSTALL_PREFIX=instDir ^
      -G "Visual Studio 16 2019" -A x64 ^
      -D NO_PTEX=1 -D NO_DOC=1 ^
      -D NO_OMP=1 -D NO_TBB=1 -D NO_CUDA=1 -D NO_OPENCL=1 -D NO_CLEW=1 ^
      -D "GLFW_LOCATION=C:\path\to\glfw" ^
      -S .

cmake --build buildDir --config Release --target install
```

### Linux

```
cmake -B buildDir \
      -D CMAKE_INSTALL_PREFIX=instDir \
      -D NO_PTEX=1 -D NO_DOC=1 \
      -D NO_OMP=1 -D NO_TBB=1 -D NO_CUDA=1 -D NO_OPENCL=1 -D NO_CLEW=1 \
      -D "GLFW_LOCATION=/path/to/glfw" \
      -S .

cmake --build buildDir --config Release --target install
```

### macOS

```
cmake -B buildDir \
      -D CMAKE_INSTALL_PREFIX=instDir \
      -G Xcode \
      -D NO_PTEX=1 -D NO_DOC=1 \
      -D NO_OMP=1 -D NO_TBB=1 -D NO_CUDA=1 -D NO_OPENCL=1 -D NO_CLEW=1 \
      -D "GLFW_LOCATION=/path/to/glfw" \
      -S .

cmake --build buildDir --config Release --target install
```

### iOS

Use CMAKE_SYSTEM_NAME to have CMake use the appropriate cross-compilation toolchain when building for iOS.

```
SDKROOT=$(xcrun --sdk iphoneos --show-sdk-path) \
cmake -B buildDir \
      -D CMAKE_INSTALL_PREFIX=instDir \
      -G Xcode \
      -D CMAKE_SYSTEM_NAME=iOS \
      -D NO_PTEX=1 -D NO_DOC=1 \
      -D NO_OMP=1 -D NO_TBB=1 -D NO_CUDA=1 -D NO_OPENCL=1 -D NO_CLEW=1 \
      -D NO_TUTORIALS=1 -D NO_EXAMPLES=1 -D NO_REGRESSION=1 -D NO_OPENGL=1 \
      -S .
```

  * This will produce an "OpenSubdiv.xcodeproj" that can be opened with Xcode.
  * Use `SDKROOT=$(xcrun --sdk iphonesimulator --show-sdk-path)` instead for an iOS-Sim target

### Useful cmake options and environment variables

````
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
````

