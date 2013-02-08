#!/usr/bin/env python

#
#     Copyright (C) Pixar. All rights reserved.
#
#     This license governs use of the accompanying software. If you
#     use the software, you accept this license. If you do not accept
#     the license, do not use the software.
#
#     1. Definitions
#     The terms "reproduce," "reproduction," "derivative works," and
#     "distribution" have the same meaning here as under U.S.
#     copyright law.  A "contribution" is the original software, or
#     any additions or changes to the software.
#     A "contributor" is any person or entity that distributes its
#     contribution under this license.
#     "Licensed patents" are a contributor's patent claims that read
#     directly on its contribution.
#
#     2. Grant of Rights
#     (A) Copyright Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free copyright license to reproduce its contribution,
#     prepare derivative works of its contribution, and distribute
#     its contribution or any derivative works that you create.
#     (B) Patent Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free license under its licensed patents to make, have
#     made, use, sell, offer for sale, import, and/or otherwise
#     dispose of its contribution in the software or derivative works
#     of the contribution in the software.
#
#     3. Conditions and Limitations
#     (A) No Trademark License- This license does not grant you
#     rights to use any contributor's name, logo, or trademarks.
#     (B) If you bring a patent claim against any contributor over
#     patents that you claim are infringed by the software, your
#     patent license from such contributor to the software ends
#     automatically.
#     (C) If you distribute any portion of the software, you must
#     retain all copyright, patent, trademark, and attribution
#     notices that are present in the software.
#     (D) If you distribute any portion of the software in source
#     code form, you may do so only under this license by including a
#     complete copy of this license with your distribution. If you
#     distribute any portion of the software in compiled or object
#     code form, you may only do so under a license that complies
#     with this license.
#     (E) The software is licensed "as-is." You bear the risk of
#     using it. The contributors give no express warranties,
#     guarantees or conditions. You may have additional consumer
#     rights under your local laws which this license cannot change.
#     To the extent permitted under your local laws, the contributors
#     exclude the implied warranties of merchantability, fitness for
#     a particular purpose and non-infringement.
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
