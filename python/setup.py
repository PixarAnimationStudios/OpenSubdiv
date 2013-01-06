#!/usr/bin/env python

from distutils.core import setup, Command, Extension
import numpy
import os, os.path

# You may need to change this location,
# depending on where you built the core
# OpenSubdiv library:
osd_lib_path = '../build/lib'

def import_build_folder():
    import sys, distutils.util, os.path
    build_dir = "build/lib.{0}-{1}.{2}".format(
        distutils.util.get_platform(),
        *sys.version_info)
    if not os.path.exists(build_dir):
        print "Folder does not exist: " + build_dir
        print "Perhaps you need to run:"
        print "    python setup.py build"
    else:
        sys.path.insert(0, build_dir)

class TestCommand(Command):
    description = "runs unit tests"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import_build_folder()
        import unittest, test
        suite = unittest.defaultTestLoader.loadTestsFromModule(test)
        unittest.TextTestRunner(verbosity=2).run(suite)

class DemoCommand(Command):
    description = "runs a little PyQt demo of the Python wrapper"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import_build_folder()
        import demo
        demo.main()

class DocCommand(Command):
    description = "Generate HTML documentation with Sphinx"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import os
        os.chdir('doc')
        os.system('make clean html')

np_include_dir = numpy.get_include()
np_library_dir = os.path.join(np_include_dir, '../lib')

osd_shim = Extension('osd.shim',
                     runtime_library_dirs = [osd_lib_path],
                     include_dirs = ['../opensubdiv', '../regression'],
                     library_dirs = ['../build/lib', np_library_dir],
                     libraries = ['osdCPU', 'npymath'],
                     sources = ['osd/shim.cpp'])

# Disable warnings produced by numpy headers:
osd_shim.extra_compile_args = [
    "-Wno-unused-function"]

os.environ['ARCHFLAGS'] = '-arch ' + os.uname()[4]

setup(name = "OpenSubdiv",
      version = "0.1",
      packages = ['osd'],
      author = 'Pixar Animation Studios',
      cmdclass = {'test': TestCommand,
                  'doc':  DocCommand,
                  'demo': DemoCommand},
      include_dirs = [np_include_dir], 
      ext_modules = [osd_shim],
      description = 'Python Bindings to the Pixar Subdivision Library')
