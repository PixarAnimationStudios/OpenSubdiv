# OpenSubdiv

OpenSubdiv is a set of open source libraries that implement high performance subdivision surface (subdiv) evaluation on massively parallel CPU and GPU architectures. This codepath is optimized for drawing deforming subdivs with static topology at interactive framerates. The resulting limit surface matches Pixar's Renderman to numerical precision.

OpenSubdiv is covered by the Apache license, and is free to use for commercial or non-commercial use. This is the same code that Pixar uses internally for animated film production. Our intent is to encourage high performance accurate subdiv drawing by giving away the "good stuff".

Feel free to use it and let us know what you think.

For more details about OpenSubdiv, see [Pixar Graphics Technologies](http://graphics.pixar.com).

## Git Flow

We have adopted the git flow branching model. It is not necessary to use the git-flow extensions, though you may find them useful! But it will be helpful to read about the git flow branching model in order to understand the organization of branches and tags that you will find in the repository.

* [Getting Started](http://graphics.pixar.com/opensubdiv/docs/getting_started.html)
* [git-flow extensions](https://github.com/nvie/gitflow) 

## Quickstart

 * Clone
 * Make a subdirectory "build" and cd into it
 * Run cmake .. followed up with your build tool of choice (make or an IDE).

Additional detailed instructions can be found in the documentation:

 * [Getting Started](http://graphics.pixar.com/opensubdiv/docs/getting_started.html)
 * [Building OpenSubdiv](http://graphics.pixar.com/opensubdiv/docs/cmake_build.html)

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
cmake -DNO_CUDA=1 -DCMAKE_TOOLCHAIN_FILE=[path to iOS.cmake] -GXcode ..

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

