#!/usr/bin/env python
#
#     Copyright 2013 Pixar
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License
#     and the following modification to it: Section 6 Trademarks.
#     deleted and replaced with:
#
#     6. Trademarks. This License does not grant permission to use the
#     trade names, trademarks, service marks, or product names of the
#     Licensor and its affiliates, except as required for reproducing
#     the content of the NOTICE file.
#
#     You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing,
#     software distributed under the License is distributed on an
#     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
#     either express or implied.  See the License for the specific
#     language governing permissions and limitations under the
#     License.
#

from distutils.core import setup, Command, Extension
from distutils.command.build import build

import numpy
import os, os.path

np_include_dir = numpy.get_include()
np_library_dir = os.path.join(np_include_dir, '../lib')
osd_include_dirs = ['../opensubdiv', '../regression']
osddir = '../build/lib'

osd_shim = Extension(
    'osd._shim',
    include_dirs = osd_include_dirs,
    library_dirs = [osddir, np_library_dir],
    libraries = ['osdCPU', 'npymath'],
    swig_opts = ['-c++'],
    sources = [
        'osd/osdshim.i',
        'osd/subdivider.cpp',
        'osd/topology.cpp'])

osd_shim.extra_compile_args = \
    ["-Wno-unused-function"]

os.environ['ARCHFLAGS'] = '-arch ' + os.uname()[4]

def setBuildFolder(folder):
    osddir = folder
    osd_shim.runtime_library_dirs = [folder]
    osd_shim.library_dirs = [folder, np_library_dir]

def setCompilerFlags(flags):
    osd_shim.extra_compile_args = flags.split() + osd_shim.extra_compile_args

def importBuildFolder():
    import os.path
    builddir = os.path.join(osddir, "../python")
    if not os.path.exists(builddir):
        print "Folder does not exist: " + builddir
        print "Perhaps you need to run:"
        print "    python setup.py build"
    else:
        sys.path.insert(0, builddir)

class TestCommand(Command):
    description = "runs unit tests"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        importBuildFolder()
        import unittest, test
        suite = unittest.defaultTestLoader.loadTestsFromModule(test)
        unittest.TextTestRunner(verbosity=2).run(suite)

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

class BuildCommand(build):
    description = "Builds the Python bindings"
    user_options = build.user_options[:]
    user_options.extend([('osddir=', 'o', 'directory that contains libosdCPU.a etc')])
    user_options.extend([('cxxflags=','c', 'compiler flags')])
    user_options.extend([('swigopts=','s', 'swig command options')])
    def initialize_options(self):
        build.initialize_options(self)
        self.osddir = None
        self.cxxflags = None
        self.swigopts = None
    def finalize_options(self):
        build.finalize_options(self)
        if self.osddir is None:
            self.osddir = '../build/lib'
        setBuildFolder(self.osddir)
        if self.cxxflags is None:
            self.cxxflags = [(-Wall)]
        setCompilerFlags(self.cxxflags)
        if self.swigopts:
            osd_shim.swig_opts+=[self.swigopts]
    def run(self):
        build.run(self)

setup(name = "OpenSubdiv",
      version = "0.1",
      packages = ['osd'],
      author = 'Pixar Animation Studios',
      cmdclass = {
        'build': BuildCommand,
        'test': TestCommand,
        'doc':  DocCommand},
      include_dirs = [np_include_dir], 
      ext_modules = [osd_shim],
      description = 'Python Bindings to the Pixar Subdivision Library')
