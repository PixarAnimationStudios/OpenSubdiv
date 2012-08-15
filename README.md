# OpenSubdiv

OpenSubdiv is a set of open source libraries that implement high performance subdivision surface (subdiv) evaluation on massively parallel CPU and GPU architectures. This codepath is optimized for drawing deforming subdivs with static topology at interactive framerates. The resulting limit surface matches Pixar’s Renderman to numerical precision.

OpenSubdiv is covered by the [Microsoft Public License](http://www.microsoft.com/en-us/openness/licenses.aspx#MPL), and is free to use for commercial or non-commercial use. This is the same code that Pixar uses internally for animated film production. Our intent is to encourage high performance accurate subdiv drawing by giving away the "good stuff".

OpenSubdiv is entering open beta for [SIGGRAPH 2012](http://s2012.siggraph.org/). Feel free to use it and let us know what you think.

For more details about OpenSubdiv, see [Pixar Graphics Technologies](http://graphics.pixar.com).

Note that this beta code is live and will undergo significant churn as it approaches release. Expect that APIs will change as the code is continually improved.


## Quickstart

Basic instructions to get started with the code.

### Dependencies

Cmake will adapt the build based on which dependencies have been successfully discovered and will disable certain features and code examples that are being built accordingly.

Please refer to the documentation of each of the dependency packages for specific build and installation instructions.

Required:
* [cmake](http://www.cmake.org/cmake/resources/software.html)
* [GLEW](http://sourceforge.net/projects/glew/) (for now)

Optional:
* [CUDA](http://developer.nvidia.com/category/zone/cuda-zone)
* [OpenCL](http://www.khronos.org/opencl/)
* [GLUT](http://freeglut.sourceforge.net/)
* [Ptex](https://github.com/wdas/ptex)
* [Zlib](http://www.zlib.net) (required for Ptex under Windows)
* [Maya SDK](http://www.autodesk.com/maya/) (sample code for Maya viewport 2.0 primitive)

### Useful cmake options and environment variables

````
-DCMAKE_BUILD_TYPE=[Debug|Release]
-DCUDA_TOOLKIT_ROOT_DIR=[path to CUDA]
-DPTEX_LOCATION=[path to Ptex]
-DGLEW_LOCATION=[path to GLEW]
-DGLUT_LOCATION=[path to GLUT]
-DMAYA_LOCATION=[path to Maya]
````

The paths to Maya, Ptex, GLUT, and GLEW can also be specified through the
following environment variables: `MAYA_LOCATION`, `PTEX_LOCATION`, `GLUT_LOCATION`,
and `GLEW_LOCATION`.


### Build instructions for linux:

__Clone the repository:__

````
git clone git://github.com/PixarAnimationStudios/OpenSubdiv.git
````

__Generate Makefiles:__

````
cd OpenSubdiv
mkdir build
cd build
cmake ..
````

__Build the project:__

````
make
````
### Build instructions for windows:


__Clone the repository:__

In the GitShell CLI :

````
git clone git://github.com/PixarAnimationStudios/OpenSubdiv.git
````
__Generate a VC++ Solution:__

* Run cmake's GUI tool
* Set the source and build directories
* Add the location of the optional packages as variable entries
* Click 'configure' and then 'generate'

### Standalone viewer

__To run viewer:__

````
bin/viewer
````

__Usage:__

````
Left mouse button drag   : orbit camera
Middle mouse button drag : dolly camera
Right mouse button       : show popup menu
n, p                     : next/prev model
1, 2, 3, 4, 5, 6, 7      : specify subdivision level
w                        : switch display mode
q                        : quit
````

## Wish List

There are many things we'd love to do to improve support for subdivs but don't have the resources to. We hope folks feel welcome to contribute if they have the interest and time. Some things that could be improved:

  * The reference maya plugin doesn't integrate with Maya shading.  That would be cool.
  * John Lasseter loves looking at film assets in progress on an iPad. If anyone were to get this working on iOS he'd be looking at your code, and the apple geeks in all of us would smile.
  * Alembic support would be wonderful, but we don't use Alembic enough internally to do the work.
  * The precomputation step with hbr can be slow. Does anyone have thoughts on higher performance with topology rich data structures needed for feature adaptive subdivision? Maybe a class that packs adjacency into blocks of indices efficiently, or supports multithreading ?
