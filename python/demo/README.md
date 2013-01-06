This folder defines a small demo application that uses PyQt and newish version of PyOpenGL.

![Screenshot](http://raw.github.com/prideout/OpenSubdiv/master/python/demo/screenshot.png)

- **main.py**         All calls to the OSD wrapper go here.  This creates a `QApplication` and periodically pushes new VBO data into the renderer. (see below)
- **demo.py**         Defines the renderer; implements `draw` and `init`.  All OpenGL calls are made in this file, and there's no dependency on Qt or OSD.
- **canvas.py**       Inherits from `QGLWidget` and calls out to the renderer object (see above)
- **shaders.py**      Implements a miniature FX format by extracting named strings from a file and pasting them together
- **simple.glsl**     Specifies the GLSL shaders for the demo using the miniature FX format
- **utility.py**      Some linear algebra stuff to make it easier to use Modern OpenGL
- **window.py**       Inherits from `QMainWindow`, instances a canvas object
- **\_\_init\_\_.py** Exports `main` into the package namespace to make it easy to run the demo from `setup.py`
