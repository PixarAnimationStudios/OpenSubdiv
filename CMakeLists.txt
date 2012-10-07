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

project(OpenSubdiv)

cmake_minimum_required(VERSION 2.8)

set(OpenSubdiv_VERSION_MAJOR 0)
set(OpenSubdiv_VERSION_MINOR 1)
set(OpenSubdiv_VERSION_PATCH 0)

set(OpenSubdiv_VERSION
    ${OpenSubdiv_VERSION_MAJOR}.${OpenSubdiv_VERSION_MINOR}.${OpenSubdiv_VERSION_PATCH})

message(STATUS "Compiling ${CMAKE_PROJECT_NAME} version ${OpenSubdiv_VERSION}")
message(STATUS "Using cmake version ${CMAKE_VERSION}")

# Specify the default install path
SET( CMAKE_INSTALL_PREFIX
     ${PROJECT_BINARY_DIR}/ )

# Set the directory where the executables will be stored.
set(EXECUTABLE_OUTPUT_PATH
    ${PROJECT_BINARY_DIR}/bin
    CACHE PATH
    "Directory where executables will be stored"
)

# Set the directory where the libraries will be stored.
set(LIBRARY_OUTPUT_PATH
    ${PROJECT_BINARY_DIR}/lib
    CACHE PATH
    "Directory where all libraries will be stored"
)

# Specify the list of directories to search for cmake modules.
set(CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

# Ignore rules that will re-run cmake (this will avoid constant
# reloading of the generated Visual Studio project).
set(CMAKE_SUPPRESS_REGENERATION TRUE)

# Check for dependencies
find_package(OpenMP)
find_package(OpenGL)
find_package(OpenCL)
find_package(CUDA)
find_package(GLUT)
find_package(PTex)

if (NOT APPLE)
    find_package(GLEW REQUIRED)
endif()

find_package(Maya)

# Warn about missing dependencies that will cause parts of OpenSubdiv to be
# disabled.  Also, add preprocessor defines that can be used in the source
# code to determine if a specific dependency is present or not.

if(OPENMP_FOUND)
    add_definitions(
        -DOPENSUBDIV_HAS_OPENMP
        ${OpenMP_CXX_FLAGS}
    )
else()
    message(WARNING
        "OpenMP was not found : support for OMP parallel compute kernels "
        "will be diabled in Osd.  If your compiler supports OpenMP "
        "directives, please refer to the FindOpenMP.cmake shared module "
        "in your cmake installation.")
endif()

# note : (GLSL compute kernels require GL 4.2, which currently excludes APPLE)
if(OPENGL_FOUND AND GLEW_FOUND AND (NOT APPLE))
    add_definitions(
        -DOPENSUBDIV_HAS_GLSL
    )
else()
    message(WARNING
        "OpenGL was not found : support for GLSL parallel compute kernels "
        "will be disabled in Osd.  If you have an OpenGL SDK installed "
        "(version 4.2 or above), please refer to the FindOpenGL.cmake "
        "shared module in your cmake installation.")
endif()

if(OPENCL_FOUND)
    add_definitions(
        -DOPENSUBDIV_HAS_OPENCL
    )
else()
    message(WARNING
        "OpenCL was not found : support for OpenCL parallel compute kernels "
        "will be disabled in Osd.  If you have the OpenCL SDK installed, "
        "please refer to the FindOpenCL.cmake in ${PROJECT_SOURCE_DIR}/cmake.")
endif()

if(CUDA_FOUND)
    add_definitions(
        -DOPENSUBDIV_HAS_CUDA
    )
else()
    message(WARNING
        "CUDA was not found : support for CUDA parallel compute kernels "
        "will be disabled in Osd.  If you have the CUDA SDK installed, please "
        "refer to the FindCUDA.cmake shared module in your cmake installation.")
endif()

if(PTEX_FOUND)
    add_definitions(
        -DOPENSUBDIV_HAS_PTEX
    )
else()
    message(WARNING
        "Ptex was not found : the OpenSubdiv Ptex example will not be "
        "available.  If you do have Ptex installed and see this message, "
        "please add your Ptex path to FindPTex.cmake in "
        "${PROJECT_SOURCE_DIR}/cmake or set it through the PTEX_LOCATION "
        "cmake command line argument or environment variable."
    )
endif()

if(MAYA_FOUND)
    add_definitions(
        -DOPENSUBDIV_HAS_MAYA
    )
else()
    message(WARNING
        "Maya was not found : the OpenSubdiv mayaViwer plugin will not be "
        "available.  If you do have Maya installed and see this message, "
        "please add your Maya path to FindMaya.cmake in "
        "${PROJECT_SOURCE_DIR}/cmake or set it through the MAYA_LOCATION "
        "cmake command line argument or environment variable."
    )
endif()

# Link examples & regressions dynamically against Osd
set( OSD_LINK_TARGET osd_dynamic )

if (WIN32)
    add_definitions(
        # Link against the static version of GLEW.
        -DGLEW_STATIC
    )
    # Link examples & regressions statically against Osd for
    # Windows until all the kinks can be worked out.
    set( OSD_LINK_TARGET osd_static )
endif()

if(MSVC)
    # MSVC is unfortunately not standard conforming with regards to
    # the alternative names for logical and bitwise operators:
    # http://stackoverflow.com/questions/555505/c-alternative-tokens
    # http://stackoverflow.com/questions/6006526/c-writing-or-instead-of
    #
    # This can be solved by including iso646.h, but that is a rather
    # unsatisfactory solution since we then always have to remember to
    # include this header file.  Instead we define these operators
    # ourselves as command line arguments to cl.exe.
    #
    # An alternative would be to compile with the /Za option
    # (but unfortunately that breaks other code):
    # http://msdn.microsoft.com/en-us/library/0k0w269d.aspx
    add_definitions(
        /Dand=&&
        /Dand_eq=&=
        /Dbitand=&
        /Dbitor=|
        /Dcompl=~
        /Dnot=!
        /Dnot_eq=!=
        /Dor=||
        /Dor_eq=|=
        # nvcc does not seem to like a caret being the last character
        # in a command line defined preprocessor symbol, so add an
        # empty trailing comment to avoid this.
        /Dxor=^/**/
        /Dxor_eq=^=
    )

    add_definitions(
        /W3     # Use warning level recommended for production purposes.
        /WX     # Treat all compiler warnings as errors.

        # Make sure WinDef.h does not define min and max macros which
        # will conflict with std::min() and std::max().
        /DNOMINMAX

        # Make sure the constants in <math.h> get defined.
        /D_USE_MATH_DEFINES

        # Do not enforce MSVC's safe CRT replacements.
        /D_CRT_SECURE_NO_WARNINGS

        # Disable checked iterators and iterator debugging.  Visual Studio
        # 2008 does not implement std::vector::data(), so we need to take the
        # address of std::vector::operator[](0) to get the memory location of
        # a vector's underlying data storage.  This does not work for an empty
        # vector if checked iterators or iterator debugging is enabled.
        /D_SECURE_SCL=0
        /D_HAS_ITERATOR_DEBUGGING=0
    )
endif(MSVC)



# Macro for adding a cuda executable if cuda is found and a regular
# executable otherwise.
macro(_add_possibly_cuda_executable target)
    if(CUDA_FOUND)
        cuda_add_executable(${target} ${ARGN})
    else()
        add_executable(${target} ${ARGN})
    endif()
endmacro()


# Macro for adding a cuda library if cuda is found and a regular
# library otherwise.
macro(_add_possibly_cuda_library target)
    if(CUDA_FOUND)
        cuda_add_library(${target} ${ARGN})
    else()
        add_library(${target} ${ARGN})
    endif()
endmacro()


# Macro for adding a (potentially cuda) executable.
macro(_add_glut_executable target)

    _add_possibly_cuda_executable(${target} ${ARGN})

    if(WIN32)
        # Windows needs some of its dependency dll's copied into the same
        # directory as the executable.
        set( LIBRARIES ${GLUT_LIBRARIES})
        foreach (LIB ${LIBRARIES} )
            string(REPLACE ".lib" ".dll" DLL ${LIB})
            string(REPLACE ".LIB" ".DLL" DLL ${DLL})
            add_custom_command(
                TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${DLL}
                    $<TARGET_FILE_DIR:${target}>
            )
        endforeach()
    endif()

endmacro()


add_subdirectory(opensubdiv)

add_subdirectory(regression)

add_subdirectory(examples)

add_subdirectory(documentation)
