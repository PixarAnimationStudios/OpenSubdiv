This folder defines a small demo application that requires PyQt, PyOpenGL, and the Python bindings for OpenSubdiv (which in turn require numpy and SWIG).

![Screenshot](http://github.com/PixarAnimationStudios/OpenSubdiv/raw/master/python/doc/screenshot.png)

- **main.py**         This is what you invoke from the command line.  All calls to the `osd` module go here.  Creates a `QApplication` and periodically pushes new VBO data into the renderer. (see below)
- **renderer.py**     Defines the renderer; implements `draw` and `init`.  All OpenGL calls are made in this file, and there's no dependency on Qt or `osd`.
- **canvas.py**       Inherits from `QGLWidget` and calls out to the renderer object (see above)
- **shaders.py**      Implements a miniature FX format by extracting named strings from a file and pasting them together
- **simple.glsl**     Specifies the GLSL shaders for the demo using the miniature FX format
- **utility.py**      Some linear algebra stuff to make it easier to use Modern OpenGL
- **window.py**       Inherits from `QMainWindow`, instances a canvas object
- **interactive.py**  Invoke this from the command line to spawn an alternative demo that has an interactive prompt.
- **\_\_init\_\_.py** Exports `main` into the package namespace to make it easy to run the demo from `setup.py`
