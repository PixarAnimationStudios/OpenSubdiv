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

project(OpenSubdiv)

cmake_minimum_required(VERSION 2.8.6)

#-------------------------------------------------------------------------------
# Obtain OpenSubdiv API version from version.h file
if(EXISTS "${CMAKE_SOURCE_DIR}/opensubdiv/version.h")
    file(STRINGS "${CMAKE_SOURCE_DIR}/opensubdiv/version.h"
        OpenSubdiv_VERSION REGEX "^#define OPENSUBDIV_VERSION .*$")
    string(REPLACE "#define OPENSUBDIV_VERSION " "" OpenSubdiv_VERSION ${OpenSubdiv_VERSION})
endif()

#-------------------------------------------------------------------------------

message(STATUS "Compiling ${CMAKE_PROJECT_NAME} version ${OpenSubdiv_VERSION}")
message(STATUS "Using cmake version ${CMAKE_VERSION}")

#-------------------------------------------------------------------------------
# Specify the default install path
if (NOT DEFINED CMAKE_INSTALL_PREFIX)
    SET( CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/ )
endif()

if (NOT DEFINED CMAKE_INCDIR_BASE)
    set( CMAKE_INCDIR_BASE include/opensubdiv )
endif()

if (NOT DEFINED CMAKE_BINDIR_BASE)
    set( CMAKE_BINDIR_BASE bin )
endif()

if (NOT DEFINED CMAKE_LIBDIR_BASE)
    set( CMAKE_LIBDIR_BASE lib )
endif()

if (NOT DEFINED CMAKE_PLUGINDIR_BASE)
    set( CMAKE_PLUGINDIR_BASE plugin )
endif()

# Allow install path to be overridden for cross-compile builds
if(LIBRARY_OUTPUT_PATH_ROOT)
    SET( CMAKE_INSTALL_PREFIX ${LIBRARY_OUTPUT_PATH_ROOT}/ )
endif()

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

list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#-------------------------------------------------------------------------------
# Detect Clang (until a cmake version provides built-in variables)
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_COMPILER_IS_CLANGCC 1)
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
    set(CMAKE_COMPILER_IS_ICC 1)
endif()


set(OSD_COMPILER_FLAGS)

# Disable spurrious warnings in gcc builds and clang
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANGCC OR CMAKE_COMPILER_IS_ICC )

    # Turn on all warnings
    list(APPEND OSD_COMPILER_FLAGS -Wall)

    # HBR uses the offsetof macro on a templated struct, which appears
    # to spurriously set off this warning in both gccc and Clang
    list(APPEND OSD_COMPILER_FLAGS -Wno-invalid-offsetof)

    # HBR uses unions as an optimization for its memory allocation.
    # Type casting between union members breaks strict aliasing rules from
    # gcc 4.4.1 versions onwards. We disable the warning but keep aliasing
    # optimization.
    list(APPEND OSD_COMPILER_FLAGS -Wno-strict-aliasing)

    # FAR and OSD have templated virtual function implementations that trigger
    # a lot of hidden virtual function overloads (some of them spurrious).
    # Disable those for now in Clang.
    if(CMAKE_COMPILER_IS_CLANGCC)
        list(APPEND OSD_COMPILER_FLAGS -Wno-overloaded-virtual)
    endif()

    # Intel's icc compiler requires some libraries linked
    if(CMAKE_COMPILER_IS_ICC)

        foreach (ICC_LIB iomp5 irng intlc)

            find_library( ICC_${ICC_LIB}
                NAMES
                    ${ICC_LIB}
                PATHS
                    ${ICC_LOCATION}/lib
                    /opt/intel/lib/
                PATH_SUFFIXES
                    intel64/
                    ia32/
                    mic/
            )

            if (ICC_${ICC_LIB})
                list(APPEND ICC_LIBRARIES ${ICC_${ICC_LIB}})
            else()
                message( FATAL_ERROR "${ICC_${ICC_LIB}} library not found - required by icc" )
            endif()

        endforeach()
    endif()

elseif(MSVC)

    # Turn on all warnings
    list(APPEND OSD_COMPILER_FLAGS /Wall)

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
    list(APPEND OSD_COMPILER_FLAGS
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


    list(APPEND OSD_COMPILER_FLAGS
                    /W3     # Use warning level recommended for production purposes.
                    /WX     # Treat all compiler warnings as errors.

                    # these warnings are being triggered from inside VC's header files
                    # warning C4350: behavior change: 'member1' called instead of 'member2'
                    /wd4350
                    # warning C4548: expression before comma has no effect; expected expression with side-effect
                    /wd4548

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

                    # XXXX manuelk : we can't force SECURE_SCL to 0 or client code has
                    # problems linking against OSD if their build is not also
                    # overriding SSCL to the same value.
                    # See : http://msdn.microsoft.com/en-us/library/vstudio/hh697468.aspx
                    #/D_SECURE_SCL=0
                    #/D_HAS_ITERATOR_DEBUGGING=0
    )

    # Turn off a duplicate LIBCMT linker warning
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")

endif()

if(${SIMD} MATCHES "AVX")
    list(APPEND OSD_COMPILER_FLAGS -xAVX)
endif()

add_definitions(${OSD_COMPILER_FLAGS})

#-------------------------------------------------------------------------------

# Ignore rules that will re-run cmake (this will avoid constant
# reloading of the generated Visual Studio project).
set(CMAKE_SUPPRESS_REGENERATION TRUE)

# Check for dependencies
if(NOT NO_OMP)
    find_package(OpenMP)
endif()
if(NOT NO_TBB)
    find_package(TBB 4.0)
endif()
find_package(OpenGL)
find_package(OpenGLES)
find_package(OpenCL 1.1)
if(NOT NO_CUDA)
    find_package(CUDA 4.0)
endif()
if(NOT ANDROID AND NOT IOS)
    find_package(GLFW 2.7.0)
endif()
find_package(PTex 2.0)
find_package(PythonInterp 2.6)
find_package(SWIG 1.3.40)
find_package(Doxygen 1.8.4)
find_package(Docutils 0.6)
if (OPENGL_FOUND)
    if (APPLE)
        find_package(GLEW)
    else()
        find_package(GLEW REQUIRED)
    endif()
endif()

if (WIN32)
   find_package(DXSDK)
endif()

if (NOT NO_GCD AND APPLE)
    set(GCD_FOUND 1)
endif()

find_package(Maya 201200)

# Warn about missing dependencies that will cause parts of OpenSubdiv to be
# disabled.  Also, add preprocessor defines that can be used in the source
# code to determine if a specific dependency is present or not.

if(GCD_FOUND)
    add_definitions( -DOPENSUBDIV_HAS_GCD )
endif()

if(OPENMP_FOUND)
    add_definitions(
        -DOPENSUBDIV_HAS_OPENMP
        ${OpenMP_CXX_FLAGS}
    )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING
        "OpenMP was not found : support for OMP parallel compute kernels "
        "will be diabled in Osd.  If your compiler supports OpenMP "
        "directives, please refer to the FindOpenMP.cmake shared module "
        "in your cmake installation.")
endif()

if(TBB_FOUND)
    add_definitions(
        -DOPENSUBDIV_HAS_TBB
        ${TBB_CXX_FLAGS}
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TBB_CXX_FLAGS}")
else()
    message(WARNING
        "TBB was not found : support for TBB parallel compute kernels "
        "will be diabled in Osd.  If your compiler supports TBB "
        "directives, please refer to the FindTBB.cmake shared module "
        "in your cmake installation.")
endif()

if(GLFW_FOUND AND (GLFW_VERSION VERSION_EQUAL 3.0 OR GLFW_VERSION VERSION_GREATER 3.0))
    add_definitions( -DGLFW_VERSION_3 )
endif()

# note: perhaps rename this to be more consistent, e.g. OPENSUBDIV_USES_GLEW
if(GLEW_FOUND)
    add_definitions( -DOSD_USES_GLEW )
endif()

# note : (GLSL transform feedback kernels require GL 4.2)
if(GLEW_FOUND AND OPENGL_4_2_FOUND)
    add_definitions(
        -DOPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    )
else()
    message(WARNING
        "OpenGL 4.2 was not found : support for GLSL transform feedback kernels "
        "will be disabled in Osd.  If you have an OpenGL SDK installed "
        "(version 4.2 or above), please refer to the FindOpenGL.cmake "
        "shared module in your cmake installation.")
endif()

# note : (GLSL compute shader kernels require GL 4.3)
if(GLEW_FOUND AND OPENGL_4_3_FOUND)
    add_definitions(
        -DOPENSUBDIV_HAS_GLSL_COMPUTE
    )
else()
    message(WARNING
        "OpenGL 4.3 was not found : support for GLSL compute shader kernels "
        "will be disabled in Osd.  If you have an OpenGL SDK installed "
        "(version 4.3 or above), please refer to the FindOpenGL.cmake "
        "shared module in your cmake installation.")
endif()

if(OPENGLES_FOUND)
    add_definitions(
        -DOPENSUBDIV_HAS_OPENGLES
    )
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
set( OSD_LINK_TARGET osd_dynamic_cpu osd_dynamic_gpu )

if (WIN32)
    add_definitions(
        # Link against the static version of GLEW.
        -DGLEW_STATIC
    )
    # Link examples & regressions statically against Osd for
    # Windows until all the kinks can be worked out.
    set( OSD_LINK_TARGET osd_static_cpu osd_static_gpu )

    if (DXSDK_FOUND)
        add_definitions(
            -DOPENSUBDIV_HAS_DX11SDK
        )
    else()
        message(WARNING
            "DirectX11 SDK was not found. "
            "If you do have DXSDK installed and see this message, "
            "please add your sdk path to FindDirectX.cmake in "
            "${PROJECT_SOURCE_DIR}/cmake or set it through the "
            "DXSDK_LOCATION cmake command line argument or "
            "environment variable."
        )
    endif()
endif()


#-------------------------------------------------------------------------------
# General-use macros

# Macro for processing public headers into the build area for doxygen processing

#set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_custom_target( public_headers )

macro(_add_doxy_headers headers)
    if (NOT NO_DOC AND DOXYGEN_FOUND)
        file(RELATIVE_PATH path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )

        string(REPLACE  "/" "_" targetpath ${path})

        foreach (header ${headers})

            set(infile ${CMAKE_CURRENT_SOURCE_DIR}/${header})
            set(outfile ${CMAKE_BINARY_DIR}/public_headers/${path}/${header})
            set(targetname "${targetpath}_${header}")

            add_custom_command(
                OUTPUT
                    ${outfile}
                COMMAND
                    ${CMAKE_COMMAND}
                ARGS
                    -E copy ${infile} ${outfile}
                DEPENDS
                    ${infile}
            )

            add_custom_target(${targetname} DEPENDS ${outfile})

            #set_target_properties(${targetname} PROPERTIES FOLDER "public_headers/")

            list(APPEND headerfiles ${targetname} )
        endforeach()

        add_dependencies( public_headers DEPENDS ${headerfiles} )
    endif()
endmacro()


# Macro for adding a cuda executable if cuda is found and a regular
# executable otherwise.
macro(_add_possibly_cuda_executable target)
    if(CUDA_FOUND)
        cuda_add_executable(${target} ${ARGN})
    else()
        add_executable(${target} ${ARGN})
    endif()

    if(CMAKE_COMPILER_IS_ICC)
        target_link_libraries(${target} ${ICC_LIBRARIES})
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
macro(_add_glfw_executable target)

    _add_possibly_cuda_executable(${target} ${ARGN})

    if(WIN32)
        # Windows needs some of its dependency dll's copied into the same
        # directory as the executable.
        set( LIBRARIES ${GLFW_LIBRARIES})
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

#-------------------------------------------------------------------------------
# Build targets

add_subdirectory(opensubdiv)

if (NOT NO_REGRESSION AND NOT ANDROID AND NOT IOS) # XXXdyu
    add_subdirectory(regression)
endif()

if (NOT NO_EXAMPLES)
    add_subdirectory(examples)
endif()

if (NOT NO_PYTHON)
    add_subdirectory(python)
endif()

if (NOT NO_DOC)
    add_subdirectory(documentation)
endif()
