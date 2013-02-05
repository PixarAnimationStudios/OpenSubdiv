# Instructions

The OpenSubdiv Python wrapper has been tested with Python 2.6 and Python 2.7.
Make sure you have the numpy module installed before you begin.

First, try building the extension with:

    ./setup.py build osddir='../build/lib'

You'll need to replace `../build/lib` with the folder that has `libosdCPU.a` et al.

This creates a build folder with a platform-specific subfolder, such as:

    ./build/lib.macosx-10.8-intel-2.7

Next, try out the unit tests:

    ./setup.py test

You can clean, build, and test in one go like this:

    ./setup.py clean --all build test

If you'd like, you can try out an OpenGL demo.  For this, you need to have PyQt and PyOpenGL installed.

    ./setup.py install --user demo

You can also install the module globally with:

    sudo ./setup.py install

After installing the module, you can generate and view the Sphinx-generated documentation like so:

    ./setup.py doc
    open ./doc/_build/html/index.html

# To Do Items

- Add support for face varying data by augmenting _FaceAdapter in adapters.py
  - Exercise this in the demo by getting it down to GPU (maybe do "discard" for certain faces)
- Instead of using OsdCpuVertexBuffer, create a "NumpyCpuVertexBuffer" that wraps a numpy array
- Add an API that looks very similar to the RIB parameters for RiHierarchicalSubdiv
- Remove all the caveats that are listed in the Sphinx docs :)
