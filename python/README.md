# Instructions

The OpenSubdiv Python wrapper has been tested with Python 2.6 and Python 2.7.
Make sure you install SWIG and numpy before you begin.

CMake builds the extension like this:

    ./setup.py build --osddir='../build/lib' \
                     --build-platlib='../build/python' \
                     --build-temp='../build/temp'

If you invoke this manually, you'll need to replace `../build/lib` with the folder that has `libosdCPU.a`.

The demo that uses PyQt and PyOpenGL can be found in `../examples/python`.

You can run some unit tests like so:

    ./setup.py test

You can clean, build, and test in one go like this:

    ./setup.py clean --all build test

You can generate and view the Sphinx-generated documentation like so:

    ./setup.py doc
    open ./doc/_build/html/index.html

# To Do Items

- Add support for face varying data by augmenting _FaceAdapter in adapters.py
  - Subdivider should expose a `quads` property that allows access to `hole` etc
  - Exercise this in the demo by getting it down to GPU (maybe do "discard" for certain faces)
- Instead of using OsdCpuVertexBuffer, create a "NumpyCpuVertexBuffer" that wraps a numpy array
- Add an API that looks very similar to the RIB parameters for RiHierarchicalSubdiv
- Remove all the caveats that are listed in the Sphinx docs :)
- Sphinx documentation should be CMake-ified.
