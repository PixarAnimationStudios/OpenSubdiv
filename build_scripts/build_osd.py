#
# Copyright 2019 Pixar
#
# Licensed under the Apache License, Version 2.0 (the "Apache License")
# with the following modification; you may not use this file except in
# compliance with the Apache License and the following modification to it:
# Section 6. Trademarks. is deleted and replaced with:
#
# 6. Trademarks. This License does not grant permission to use the trade
#    names, trademarks, service marks, or product names of the Licensor
#    and its affiliates, except as required to comply with Section 4(c) of
#    the License and to reproduce the content of the NOTICE file.
#
# You may obtain a copy of the Apache License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Apache License with the above modification is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the Apache License for the specific
# language governing permissions and limitations under the Apache License.
#
from distutils.spawn import find_executable

import argparse
import contextlib
import datetime
import distutils
import fnmatch
import glob
import multiprocessing
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import urllib2
import zipfile

# Helpers for printing output
verbosity = 1

def Print(msg):
    if verbosity > 0:
        print msg

def PrintWarning(warning):
    if verbosity > 0:
        print "WARNING:", warning

def PrintStatus(status):
    if verbosity >= 1:
        print "STATUS:", status

def PrintInfo(info):
    if verbosity >= 2:
        print "INFO:", info

def PrintCommandOutput(output):
    if verbosity >= 3:
        sys.stdout.write(output)

def PrintError(error):
    print "ERROR:", error

# Helpers for determining platform
def Windows():
    return platform.system() == "Windows"
def Linux():
    return platform.system() == "Linux"
def MacOS():
    return platform.system() == "Darwin"

def GetCommandOutput(command):
    """Executes the specified command and returns output or None."""
    try:
        return subprocess.check_output(
            shlex.split(command), stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError:
        pass
    return None

def GetXcodeDeveloperDirectory():
    """Returns the active developer directory as reported by 'xcode-select -p'.
    Returns None if none is set."""
    if not MacOS():
        return None

    return GetCommandOutput("xcode-select -p")

def GetVisualStudioCompilerAndVersion():
    """Returns a tuple containing the path to the Visual Studio compiler
    and a tuple for its version, e.g. (14, 0). If the compiler is not found
    or version number cannot be determined, returns None."""
    if not Windows():
        return None

    msvcCompiler = find_executable('cl')
    if msvcCompiler:
        # VisualStudioVersion environment variable should be set by the
        # Visual Studio Command Prompt.
        match = re.search(
            "(\d+).(\d+)",
            os.environ.get("VisualStudioVersion", ""))
        if match:
            return (msvcCompiler, tuple(int(v) for v in match.groups()))
    return None

def IsVisualStudio2019OrGreater():
    VISUAL_STUDIO_2019_VERSION = (16, 0)
    msvcCompilerAndVersion = GetVisualStudioCompilerAndVersion()
    if msvcCompilerAndVersion:
        _, version = msvcCompilerAndVersion
        return version >= VISUAL_STUDIO_2019_VERSION
    return False

def IsVisualStudio2017OrGreater():
    VISUAL_STUDIO_2017_VERSION = (15, 0)
    msvcCompilerAndVersion = GetVisualStudioCompilerAndVersion()
    if msvcCompilerAndVersion:
        _, version = msvcCompilerAndVersion
        return version >= VISUAL_STUDIO_2017_VERSION
    return False

def GetCPUCount():
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 1

def Run(cmd, logCommandOutput = True):
    """Run the specified command in a subprocess."""
    PrintInfo('Running "{cmd}"'.format(cmd=cmd))

    with open("log.txt", "a") as logfile:
        logfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        logfile.write("\n")
        logfile.write(cmd)
        logfile.write("\n")

        # Let exceptions escape from subprocess calls -- higher level
        # code will handle them.
        if logCommandOutput:
            p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            while True:
                l = p.stdout.readline()
                if l != "":
                    logfile.write(l)
                    PrintCommandOutput(l)
                elif p.poll() is not None:
                    break
        else:
            p = subprocess.Popen(shlex.split(cmd))
            p.wait()

    if p.returncode != 0:
        # If verbosity >= 3, we'll have already been printing out command output
        # so no reason to print the log file again.
        if verbosity < 3:
            with open("log.txt", "r") as logfile:
                Print(logfile.read())
        raise RuntimeError("Failed to run '{cmd}'\nSee {log} for more details."
                           .format(cmd=cmd, log=os.path.abspath("log.txt")))

@contextlib.contextmanager
def CurrentWorkingDirectory(dir):
    """Context manager that sets the current working directory to the given
    directory and resets it to the original directory when closed."""
    curdir = os.getcwd()
    os.chdir(dir)
    try: yield
    finally: os.chdir(curdir)

def CopyFiles(context, src, dest):
    """Copy files like shutil.copy, but src may be a glob pattern."""
    filesToCopy = glob.glob(src)
    if not filesToCopy:
        raise RuntimeError("File(s) to copy {src} not found".format(src=src))

    instDestDir = os.path.join(context.instDir, dest)
    if not os.path.isdir(instDestDir):
        os.makedirs(instDestDir)
    for f in filesToCopy:
        PrintCommandOutput("Copying {file} to {destDir}\n"
                           .format(file=f, destDir=instDestDir))
        shutil.copy(f, instDestDir)

def CopyDirectory(context, srcDir, destDir):
    """Copy directory like shutil.copytree."""
    instDestDir = os.path.join(context.instDir, destDir)
    if os.path.isdir(instDestDir):
        shutil.rmtree(instDestDir)

    PrintCommandOutput("Copying {srcDir} to {destDir}\n"
                       .format(srcDir=srcDir, destDir=instDestDir))
    shutil.copytree(srcDir, instDestDir)

def RunCMake(context, force, extraArgs = None):
    """Invoke CMake to configure, build, and install a library whose
    source code is located in the current working directory."""
    # Create a directory for out-of-source builds in the build directory
    # using the name of the current working directory.
    srcDir = os.getcwd()
    instDir = (context.osdInstDir if srcDir == context.osdSrcDir
               else context.instDir)
    buildDir = os.path.join(context.buildDir, os.path.split(srcDir)[1])
    if force and os.path.isdir(buildDir):
        shutil.rmtree(buildDir)

    if not os.path.isdir(buildDir):
        os.makedirs(buildDir)

    generator = context.cmakeGenerator
    generatorPlatform = context.cmakeGeneratorPlatform

    # On Windows, we need to explicitly specify the generator to ensure we're
    # building a 64-bit project. (Surely there is a better way to do this?)
    if generator is None and Windows():
        if IsVisualStudio2019OrGreater():
            generator = "Visual Studio 16 2019"
            generatorPlatform = "x64"
        elif IsVisualStudio2017OrGreater():
            generator = "Visual Studio 15 2017 Win64"
        else:
            generator = "Visual Studio 14 2015 Win64"

    # On macOS default to Xcode
    if generator is None and MacOS():
        generator = "Xcode"


    multiproc = "-j{procs}"
    if Windows():
        multiproc = "/M:{procs}"

    if generator == "Xcode":
        multiproc = "-jobs {procs} -parallelizeTargets"

    multiproc = multiproc.format(procs=context.numJobs)

    # On MacOS, enable the use of @rpath for relocatable builds.
    osx_rpath = None
    if MacOS():
        osx_rpath = "-DCMAKE_MACOSX_RPATH=ON"


    # Format generator appropriately
    if generator is not None:
        generator = '-G "{gen}"'.format(gen=generator)

    if generatorPlatform is not None:
        generator += ' -A "{arch}"'.format(arch=generatorPlatform)

    with CurrentWorkingDirectory(buildDir):
        Run('cmake '
            '-DCMAKE_INSTALL_PREFIX="{instDir}" '
            '-DCMAKE_PREFIX_PATH="{depsInstDir}" '
            '{osx_rpath} '
            '{generator} '
            '{extraArgs} '
            '"{srcDir}"'
            .format(instDir=instDir,
                    depsInstDir=context.instDir,
                    srcDir=srcDir,
                    osx_rpath=(osx_rpath or ""),
                    generator=(generator or ""),
                    extraArgs=(" ".join(extraArgs) if extraArgs else "")))
        Run("cmake --build . --config Release --target install -- {jobs}"
            .format(jobs=multiproc))

def PatchFile(filename, patches):
    """Applies patches to the specified file. patches is a list of tuples
    (old string, new string)."""
    oldLines = open(filename, 'r').readlines()
    newLines = oldLines
    for (oldLine, newLine) in patches:
        newLines = [s.replace(oldLine, newLine) for s in newLines]
    if newLines != oldLines:
        PrintInfo("Patching file {filename} (original in {oldFilename})..."
                  .format(filename=filename, oldFilename=filename + ".old"))
        shutil.copy(filename, filename + ".old")
        open(filename, 'w').writelines(newLines)

def DownloadFileWithCurl(url, outputFilename):
    # Don't log command output so that curl's progress
    # meter doesn't get written to the log file.
    Run("curl {progress} -L -o {filename} {url}".format(
        progress="-#" if verbosity >= 2 else "-s",
        filename=outputFilename, url=url),
        logCommandOutput=False)

def DownloadFileWithPowershell(url, outputFilename):
    # It's important that we specify to use TLS v1.2 at least or some
    # of the downloads will fail.
    cmd = "powershell [Net.ServicePointManager]::SecurityProtocol = \
            [Net.SecurityProtocolType]::Tls12; \"(new-object \
            System.Net.WebClient).DownloadFile('{url}', '{filename}')\""\
            .format(filename=outputFilename, url=url)

    Run(cmd,logCommandOutput=False)

def DownloadFileWithUrllib(url, outputFilename):
    r = urllib2.urlopen(url)
    with open(outputFilename, "wb") as outfile:
        outfile.write(r.read())

def DownloadURL(url, context, force, dontExtract = None):
    """Download and extract the archive file at given URL to the
    source directory specified in the context.

    dontExtract may be a sequence of path prefixes that will
    be excluded when extracting the archive.

    Returns the absolute path to the directory where files have
    been extracted."""
    with CurrentWorkingDirectory(context.srcDir):
        # Extract filename from URL and see if file already exists.
        filename = url.split("/")[-1]
        if force and os.path.exists(filename):
            os.remove(filename)

        if os.path.exists(filename):
            PrintInfo("{0} already exists, skipping download"
                      .format(os.path.abspath(filename)))
        else:
            PrintInfo("Downloading {0} to {1}"
                      .format(url, os.path.abspath(filename)))

            # To work around occasional hiccups with downloading from websites
            # (SSL validation errors, etc.), retry a few times if we don't
            # succeed in downloading the file.
            maxRetries = 5
            lastError = None

            # Download to a temporary file and rename it to the expected
            # filename when complete. This ensures that incomplete downloads
            # will be retried if the script is run again.
            tmpFilename = filename + ".tmp"
            if os.path.exists(tmpFilename):
                os.remove(tmpFilename)

            for i in xrange(maxRetries):
                try:
                    context.downloader(url, tmpFilename)
                    break
                except Exception as e:
                    PrintCommandOutput("Retrying download due to error: {err}\n"
                                       .format(err=e))
                    lastError = e
            else:
                errorMsg = str(lastError)
                if "SSL: TLSV1_ALERT_PROTOCOL_VERSION" in errorMsg:
                    errorMsg += ("\n\n"
                                 "Your OS or version of Python may not support "
                                 "TLS v1.2+, which is required for downloading "
                                 "files from certain websites. This support "
                                 "was added in Python 2.7.9."
                                 "\n\n"
                                 "You can use curl to download dependencies "
                                 "by installing it in your PATH and re-running "
                                 "this script.")
                raise RuntimeError("Failed to download {url}: {err}"
                                   .format(url=url, err=errorMsg))

            shutil.move(tmpFilename, filename)

        # Open the archive and retrieve the name of the top-most directory.
        # This assumes the archive contains a single directory with all
        # of the contents beneath it.
        archive = None
        rootDir = None
        members = None
        try:
            if tarfile.is_tarfile(filename):
                archive = tarfile.open(filename)
                rootDir = archive.getnames()[0].split('/')[0]
                if dontExtract != None:
                    members = (m for m in archive.getmembers()
                               if not any((fnmatch.fnmatch(m.name, p)
                                           for p in dontExtract)))
            elif zipfile.is_zipfile(filename):
                archive = zipfile.ZipFile(filename)
                rootDir = archive.namelist()[0].split('/')[0]
                if dontExtract != None:
                    members = (m for m in archive.getnames()
                               if not any((fnmatch.fnmatch(m, p)
                                           for p in dontExtract)))
            else:
                raise RuntimeError("unrecognized archive file type")

            with archive:
                extractedPath = os.path.abspath(rootDir)
                if force and os.path.isdir(extractedPath):
                    shutil.rmtree(extractedPath)

                if os.path.isdir(extractedPath):
                    PrintInfo("Directory {0} already exists, skipping extract"
                              .format(extractedPath))
                else:
                    PrintInfo("Extracting archive to {0}".format(extractedPath))

                    # Extract to a temporary directory then move the contents
                    # to the expected location when complete. This ensures that
                    # incomplete extracts will be retried if the script is run
                    # again.
                    tmpExtractedPath = os.path.abspath("extract_dir")
                    if os.path.isdir(tmpExtractedPath):
                        shutil.rmtree(tmpExtractedPath)

                    archive.extractall(tmpExtractedPath, members=members)

                    shutil.move(os.path.join(tmpExtractedPath, rootDir),
                                extractedPath)
                    shutil.rmtree(tmpExtractedPath)

                return extractedPath
        except Exception as e:
            # If extraction failed for whatever reason, assume the
            # archive file was bad and move it aside so that re-running
            # the script will try downloading and extracting again.
            shutil.move(filename, filename + ".bad")
            raise RuntimeError("Failed to extract archive {filename}: {err}"
                               .format(filename=filename, err=e))

############################################################
# 3rd-Party Dependencies

AllDependencies = list()
AllDependenciesByName = dict()

class Dependency(object):
    def __init__(self, name, installer, *files):
        self.name = name
        self.installer = installer
        self.filesToCheck = files

        AllDependencies.append(self)
        AllDependenciesByName.setdefault(name.lower(), self)

    def Exists(self, context):
        return all([os.path.isfile(os.path.join(context.instDir, f))
                    for f in self.filesToCheck])


############################################################
# Intel TBB

if Windows():
    TBB_URL = "https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_win.zip"
elif MacOS():
    TBB_URL = "https://github.com/01org/tbb/archive/2017_U2.tar.gz"
else:
    TBB_URL = "https://github.com/01org/tbb/archive/4.4.6.tar.gz"

def InstallTBB(context, force, buildArgs):
    if Windows():
        InstallTBB_Windows(context, force, buildArgs)
    elif Linux() or MacOS():
        InstallTBB_LinuxOrMacOS(context, force, buildArgs)

def InstallTBB_Windows(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(TBB_URL, context, force)):
        # On Windows, we simply copy headers and pre-built DLLs to
        # the appropriate location.

        if buildArgs:
            PrintWarning("Ignoring build arguments {}, TBB is "
                         "not built from source on this platform."
                         .format(buildArgs))

        CopyFiles(context, "bin\\intel64\\vc14\\*.*", "bin")
        CopyFiles(context, "lib\\intel64\\vc14\\*.*", "lib")
        CopyDirectory(context, "include\\serial", "include\\serial")
        CopyDirectory(context, "include\\tbb", "include\\tbb")

def InstallTBB_LinuxOrMacOS(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(TBB_URL, context, force)):
        # TBB does not support out-of-source builds in a custom location.
        Run('make -j{procs} {buildArgs}'
            .format(procs=context.numJobs,
                    buildArgs=" ".join(buildArgs)))

        CopyFiles(context, "build/*_release/libtbb*.*", "lib")
        CopyDirectory(context, "include/serial", "include/serial")
        CopyDirectory(context, "include/tbb", "include/tbb")

TBB = Dependency("TBB", InstallTBB, "include/tbb/tbb.h")


############################################################
# GLFW

GLFW_URL = "https://github.com/glfw/glfw/archive/3.2.1.zip"

def InstallGLFW(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(GLFW_URL, context, force)):
        RunCMake(context, force, buildArgs)

GLFW = Dependency("GLFW", InstallGLFW, "include/GLFW/glfw3.h")

############################################################
# zlib

ZLIB_URL = "https://github.com/madler/zlib/archive/v1.2.11.zip"

def InstallZlib(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(ZLIB_URL, context, force)):
        RunCMake(context, force, buildArgs)

ZLIB = Dependency("zlib", InstallZlib, "include/zlib.h")

############################################################
# Ptex

PTEX_URL = "https://github.com/wdas/ptex/archive/v2.1.28.zip"

def InstallPtex(context, force, buildArgs):
    if Windows():
        InstallPtex_Windows(context, force, buildArgs)
    else:
        InstallPtex_LinuxOrMacOS(context, force, buildArgs)

def InstallPtex_Windows(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(PTEX_URL, context, force)):
        # Ptex has a bug where the import library for the dynamic library and
        # the static library both get the same name, Ptex.lib, and as a
        # result one clobbers the other. We hack the appropriate CMake
        # file to prevent that. Since we don't need the static library we'll
        # rename that.
        #
        # In addition src\tests\CMakeLists.txt adds -DPTEX_STATIC to the
        # compiler but links tests against the dynamic library, causing the
        # links to fail. We patch the file to not add the -DPTEX_STATIC
        PatchFile('src\\ptex\\CMakeLists.txt',
                  [("set_target_properties(Ptex_static PROPERTIES OUTPUT_NAME Ptex)",
                    "set_target_properties(Ptex_static PROPERTIES OUTPUT_NAME Ptexs)")])
        PatchFile('src\\tests\\CMakeLists.txt',
                  [("add_definitions(-DPTEX_STATIC)",
                    "# add_definitions(-DPTEX_STATIC)")])

        RunCMake(context, force, buildArgs)

def InstallPtex_LinuxOrMacOS(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(PTEX_URL, context, force)):
        RunCMake(context, force, buildArgs)

PTEX = Dependency("Ptex", InstallPtex, "include/PtexVersion.h")

############################################################
# OpenSubdiv

def InstallOpenSubdiv(context, force, buildArgs):

    with CurrentWorkingDirectory(context.osdSrcDir):
        extraArgs = []

        if context.buildDocs:
            extraArgs.append('-DNO_DOC=OFF')
        else:
            extraArgs.append('-DNO_DOC=ON')

        if context.buildPtex:
            extraArgs.append('-DNO_PTEX=OFF')
        else:
            extraArgs.append('-DNO_PTEX=ON')

        if context.buildTBB:
            extraArgs.append('-DNO_TBB=OFF')
        else:
            extraArgs.append('-DNO_TBB=ON')

        if context.buildTests:
            extraArgs.append('-DNO_REGRESSION=OFF')
            extraArgs.append('-DNO_TESTS=OFF')
        else:
            extraArgs.append('-DNO_REGRESSION=ON')
            extraArgs.append('-DNO_TESTS=ON')

        if context.buildExamples:
            extraArgs.append('-DNO_EXAMPLES=OFF')
        else:
            extraArgs.append('-DNO_EXAMPLES=ON')

        if context.buildTutorials:
            extraArgs.append('-DNO_TUTORIALS=OFF')
        else:
            extraArgs.append('-DNO_TUTORIALS=ON')

        if context.buildOMP:
            extraArgs.append('-DNO_OMP=OFF')
        else:
            extraArgs.append('-DNO_OMP=ON')

        if context.buildCUDA:
            extraArgs.append('-DNO_CUDA=OFF')
            if context.cudaLocation:
                extraArgs.append('-DCUDA_TOOLKIT_ROOT_DIR="{location}"'
                        .format(location=context.cudaLocation))
        else:
            extraArgs.append('-DNO_CUDA=ON')

        if context.buildOpenCL:
            extraArgs.append('-DNO_OPENCL=OFF')
        else:
            extraArgs.append('-DNO_OPENCL=ON')

        if context.buildDX:
            extraArgs.append('-DNO_DX=OFF')
        else:
            extraArgs.append('-DNO_DX=ON')

        # Options we haven't yet exposed:
        # NO_CLEW
        # NO_OPENGL
        # NO_METAL
        # NO_GLTESTS
        # NO_GLEW
        # NO_GLFW
        # NO_GLFW_X11

        # OpenSubdiv's FindGLFW module won't look in CMAKE_PREFIX_PATH, so
        # we need to explicitly specify GLFW_LOCATION here.
        extraArgs.append('-DGLFW_LOCATION="{instDir}"'
                         .format(instDir=context.instDir))

        # Add on any user-specified extra arguments.
        extraArgs += buildArgs

        RunCMake(context, force, extraArgs)

OPENSUBDIV = Dependency("OpenSubdiv", InstallOpenSubdiv,
                        "include/opensubdiv/version.h")

############################################################
# Install script

programDescription = """\
Installation Script for OpenSubdiv

Builds and installs OpenSubidv and 3rd-party dependencies to specified location.

- Libraries:
The following is a list of libraries that this script will download and build
as needed. These names can be used to identify libraries for various script
options, like --force or --build-args.

{libraryList}

- Downloading Libraries:
If curl or powershell (on Windows) are installed and located in PATH, they
will be used to download dependencies. Otherwise, a built-in downloader will
be used.

- Specifying Custom Build Arguments:
Users may specify custom build arguments for libraries using the --build-args
option. This values for this option must take the form <library name>,<option>.
These arguments will be passed directly to the build system for the specified
library. Multiple quotes may be needed to ensure arguments are passed on
exactly as desired. Users must ensure these arguments are suitable for the
specified library and do not conflict with other options, otherwise build
errors may occur.
""".format(
    libraryList=" ".join(sorted([d.name for d in AllDependencies])))

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=programDescription)

parser.add_argument("install_dir", type=str,
                    help="Directory where OpenSubdiv will be installed")
parser.add_argument("-n", "--dry_run", dest="dry_run", action="store_true",
                    help="Only summarize what would happen")

group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="count", default=1,
                   dest="verbosity",
                   help="Increase verbosity level (1-3)")
group.add_argument("-q", "--quiet", action="store_const", const=0,
                   dest="verbosity",
                   help="Suppress all output except for error messages")

group = parser.add_argument_group(title="Build Options")
group.add_argument("-j", "--jobs", type=int, default=GetCPUCount(),
                   help=("Number of build jobs to run in parallel. "
                         "(default: # of processors [{0}])"
                         .format(GetCPUCount())))
group.add_argument("--build", type=str,
                   help=("Build directory for OpenSubdiv and 3rd-party "
                         "dependencies (default: <install_dir>/build)"))
group.add_argument("--build-args", type=str, nargs="*", default=[],
                   help=("Custom arguments to pass to build system when "
                         "building libraries (see docs above)"))
group.add_argument("--force", type=str, action="append", dest="force_build",
                   default=[],
                   help=("Force download and build of specified library "
                         "(see docs above)"))
group.add_argument("--force-all", action="store_true",
                   help="Force download and build of all libraries")
group.add_argument("--generator", type=str,
                   help=("CMake generator to use when building libraries with "
                         "cmake"))
group.add_argument("--generatorPlatform", type=str,
                   help=("CMake generator platform if supported by generator "
                         "to use when building libraries with cmake"))

group = parser.add_argument_group(title="3rd Party Dependency Build Options")
group.add_argument("--src", type=str,
                   help=("Directory where dependencies will be downloaded "
                         "(default: <install_dir>/src)"))
group.add_argument("--inst", type=str,
                   help=("Directory where dependencies will be installed "
                         "(default: <install_dir>)"))

group = parser.add_argument_group(title="OpenSubdiv Options")

subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument("--tests", dest="build_tests", action="store_true",
                      default=False, help="Build unit tests")
subgroup.add_argument("--no-tests", dest="build_tests", action="store_false",
                      help="Do not build tests (default)")
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument("--docs", dest="build_docs", action="store_true",
                      default=False, help="Build documentation")
subgroup.add_argument("--no-docs", dest="build_docs", action="store_false",
                      help="Do not build documentation (default)")

subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument("--examples", dest="build_examples", action="store_true",
                      default=True, help="Build examples (default)")
subgroup.add_argument("--no-examples", dest="build_example",
                      action="store_false",
                      help="Do not build examples")

subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument("--tutorials", dest="build_tutorials",
                      action="store_true",
                      default=True, help="Build tutorials (default)")
subgroup.add_argument("--no-tutorials", dest="build_tutorial",
                      action="store_false",
                      help="Do not build tutorials")

subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument("--ptex", dest="build_ptex", action="store_true",
                      default=False,
                      help="Enable Ptex support")
subgroup.add_argument("--no-ptex", dest="build_ptex",
                      action="store_false",
                      help="Disable Ptex support (default)")

subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument("--tbb", dest="build_tbb", action="store_true",
                      default=False,
                      help="Enable TBB support")
subgroup.add_argument("--no-tbb", dest="build_tbb",
                      action="store_false",
                      help="Disable TBB support (default)")

subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument("--omp", dest="build_omp", action="store_true",
                      default=False,
                      help="Enable OMP support")
subgroup.add_argument("--no-omp", dest="build_omp",
                      action="store_false",
                      help="Disable OMP support (default)")

subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument("--cuda", dest="build_cuda", action="store_true",
                      default=False,
                      help="Enable CUDA support")
subgroup.add_argument("--no-cuda", dest="build_cuda",
                      action="store_false",
                      help="Disable CUDA support (default)")

group.add_argument("--cuda-location", type=str,
                   help="Directory where the CUDA SDK is installed.")

subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument("--opencl", dest="build_opencl", action="store_true",
                      default=False,
                      help="Enable OpenCL support")
subgroup.add_argument("--no-opencl", dest="build_opencl",
                      action="store_false",
                      help="Disable OpenCL support (default)")

subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument("--directx", dest="build_dx", action="store_true",
                      default=False,
                      help="Enable DirectX support")
subgroup.add_argument("--no-directx", dest="build_dx",
                      action="store_false",
                      help="Disable DirectX support (default)")


args = parser.parse_args()

class InstallContext:
    def __init__(self, args):
        # Assume the OpenSubdiv source directory is in the parent directory
        self.osdSrcDir = os.path.normpath(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

        # Directory where OpenSubdiv will be installed
        self.osdInstDir = os.path.abspath(args.install_dir)

        # Directory where dependencies will be installed
        self.instDir = (os.path.abspath(args.inst) if args.inst
                        else self.osdInstDir)

        # Directory where dependencies will be downloaded and extracted
        self.srcDir = (os.path.abspath(args.src) if args.src
                       else os.path.join(self.osdInstDir, "src"))

        # Directory where OpenSubdiv and dependencies will be built
        self.buildDir = (os.path.abspath(args.build) if args.build
                         else os.path.join(self.osdInstDir, "build"))

        # Determine which downloader to use.  The reason we don't simply
        # use urllib2 all the time is that some older versions of Python
        # don't support TLS v1.2, which is required for downloading some
        # dependencies.
        if find_executable("curl"):
            self.downloader = DownloadFileWithCurl
            self.downloaderName = "curl"
        elif Windows() and find_executable("powershell"):
            self.downloader = DownloadFileWithPowershell
            self.downloaderName = "powershell"
        else:
            self.downloader = DownloadFileWithUrllib
            self.downloaderName = "built-in"

        # CMake generator
        self.cmakeGenerator = args.generator
        self.cmakeGeneratorPlatform = args.generatorPlatform

        # Number of jobs
        self.numJobs = args.jobs
        if self.numJobs <= 0:
            raise ValueError("Number of jobs must be greater than 0")

        # Build arguments
        self.buildArgs = dict()
        for a in args.build_args:
            (depName, _, arg) = a.partition(",")
            if not depName or not arg:
                raise ValueError("Invalid argument for --build-args: {}"
                                 .format(a))
            if depName.lower() not in AllDependenciesByName:
                raise ValueError("Invalid library for --build-args: {}"
                                 .format(depName))

            self.buildArgs.setdefault(depName.lower(), []).append(arg)

        # Dependencies that are forced to be built
        self.forceBuildAll = args.force_all
        self.forceBuild = [dep.lower() for dep in args.force_build]

        # Optional components
        self.buildExamples = args.build_examples
        self.buildTutorials = args.build_tutorials
        self.buildTests = args.build_tests
        self.buildDocs = args.build_docs
        self.buildTBB = args.build_tbb
        self.buildOMP = args.build_omp
        self.buildCUDA = args.build_cuda
        self.cudaLocation = args.cuda_location
        self.buildOpenCL = args.build_opencl
        self.buildDX = args.build_dx
        self.buildPtex = args.build_ptex

    def GetBuildArguments(self, dep):
        return self.buildArgs.get(dep.name.lower(), [])

    def ForceBuildDependency(self, dep):
        return self.forceBuildAll or dep.name.lower() in self.forceBuild

try:
    context = InstallContext(args)
except Exception as e:
    PrintError(str(e))
    sys.exit(1)

verbosity = args.verbosity

# Augment PATH on Windows so that 3rd-party dependencies can find libraries
# they depend on. In particular, this is needed for building IlmBase/OpenEXR.
extraPaths = []
if Windows():
    extraPaths.append(os.path.join(context.instDir, "lib"))
    extraPaths.append(os.path.join(context.instDir, "bin"))

if extraPaths:
    paths = os.environ.get('PATH', '').split(os.pathsep) + extraPaths
    os.environ['PATH'] = os.pathsep.join(paths)

# Determine list of dependencies that are required based on options
# user has selected.
requiredDependencies = [GLFW]

if context.buildPtex:
    # Assume zlib already exists on Linux platforms and don't build
    # our own. This avoids potential issues where a host application
    # loads an older version of zlib than the one we'd build and link
    # our libraries against.
    if not Linux():
        requiredDependencies += [ZLIB]
    requiredDependencies += [PTEX]

if context.buildTBB:
    requiredDependencies += [TBB]



dependenciesToBuild = []
for dep in requiredDependencies:
    if context.ForceBuildDependency(dep) or not dep.Exists(context):
        if dep not in dependenciesToBuild:
            dependenciesToBuild.append(dep)

# Verify toolchain needed to build required dependencies
if (not find_executable("g++") and
    not find_executable("clang") and
    not GetXcodeDeveloperDirectory() and
    not GetVisualStudioCompilerAndVersion()):
    PrintError("C++ compiler not found -- please install a compiler")
    sys.exit(1)

if not find_executable("cmake"):
    PrintError("CMake not found -- please install it and adjust your PATH")
    sys.exit(1)

if context.buildDocs:
    if not find_executable("doxygen"):
        PrintError("doxygen not found -- please install it and adjust your PATH")
        sys.exit(1)

    if not find_executable("dot"):
        PrintError("dot not found -- please install graphviz and adjust your "
                   "PATH")
        sys.exit(1)


# Summarize
summaryMsg = """
Building with settings:
  OpenSubdiv source directory   {osdSrcDir}
  OpenSubdiv install directory  {osdInstDir}
  3rd-party source directory    {srcDir}
  3rd-party install directory   {instDir}
  Build directory               {buildDir}
  CMake generator               {cmakeGenerator}
  Downloader                    {downloader}

  Building
      TBB support:              {buildTBB}
      OMP support:              {buildOMP}
      CUDA support:             {buildCUDA}
      OpenCL support:           {buildOpenCL}
      DirectX support:          {buildDX}
      Ptex support:             {buildPtex}
      Documentation:            {buildDocs}
      Examples:                 {buildExamples}
      Tutorials:                {buildTutorials}
      Tests:                    {buildTests}

  Dependencies                  {dependencies}"""

if context.buildArgs:
    summaryMsg += """
  Build arguments               {buildArgs}"""

def FormatBuildArguments(buildArgs):
    s = ""
    for depName in sorted(buildArgs.iterkeys()):
        args = buildArgs[depName]
        s += """
                                {name}: {args}""".format(
            name=AllDependenciesByName[depName].name,
            args=" ".join(args))
    return s.lstrip()

summaryMsg = summaryMsg.format(
    osdSrcDir=context.osdSrcDir,
    osdInstDir=context.osdInstDir,
    srcDir=context.srcDir,
    buildDir=context.buildDir,
    instDir=context.instDir,
    cmakeGenerator=("Default" if not context.cmakeGenerator
                    else context.cmakeGenerator),
    downloader=(context.downloaderName),
    dependencies=("None" if not dependenciesToBuild else
                  ", ".join([d.name for d in dependenciesToBuild])),
    buildArgs=FormatBuildArguments(context.buildArgs),
    buildTBB=("On" if context.buildTBB else "Off"),
    buildOMP=("On" if context.buildOMP else "Off"),
    buildCUDA=("On" if context.buildCUDA else "Off"),
    buildOpenCL=("On" if context.buildOpenCL else "Off"),
    buildDX=("On" if context.buildDX else "Off"),
    buildPtex=("On" if context.buildPtex else "Off"),
    buildDocs=("On" if context.buildDocs else "Off"),
    buildExamples=("On" if context.buildExamples else "Off"),
    buildTutorials=("On" if context.buildTutorials else "Off"),
    buildTests=("On" if context.buildTests else "Off"))

Print(summaryMsg)

if args.dry_run:
    sys.exit(0)

# Ensure directory structure is created and is writable.
for dir in [context.osdInstDir, context.instDir, context.srcDir,
            context.buildDir]:
    try:
        if os.path.isdir(dir):
            testFile = os.path.join(dir, "canwrite")
            open(testFile, "w").close()
            os.remove(testFile)
        else:
            os.makedirs(dir)
    except Exception as e:
        PrintError("Could not write to directory {dir}. Change permissions "
                   "or choose a different location to install to."
                   .format(dir=dir))
        sys.exit(1)

try:
    # Download and install 3rd-party dependencies, followed by OpenSubdiv.
    for dep in dependenciesToBuild + [OPENSUBDIV]:
        PrintStatus("Installing {dep}...".format(dep=dep.name))
        dep.installer(context,
                      buildArgs=context.GetBuildArguments(dep),
                      force=context.ForceBuildDependency(dep))
except Exception as e:
    PrintError(str(e))
    sys.exit(1)

requiredInPath = set([
    os.path.join(context.osdInstDir, "bin")
])
requiredInPath.update(extraPaths)

if Windows():
    requiredInPath.update([
        os.path.join(context.osdInstDir, "lib"),
        os.path.join(context.instDir, "bin"),
        os.path.join(context.instDir, "lib")
    ])

Print("""
Success! To use OpenSubdiv, please ensure that you have:""")

Print("""
    The following in your PATH environment variable:
    {requiredInPath}
""".format(requiredInPath="\n    ".join(sorted(requiredInPath))))
