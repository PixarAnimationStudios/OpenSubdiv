#
#   Copyright 2013 Pixar
#
#   Licensed under the Apache License, Version 2.0 (the "Apache License")
#   with the following modification; you may not use this file except in
#   compliance with the Apache License and the following modification to it:
#   Section 6. Trademarks. is deleted and replaced with:
#
#   6. Trademarks. This License does not grant permission to use the trade
#      names, trademarks, service marks, or product names of the Licensor
#      and its affiliates, except as required to comply with Section 4(c) of
#      the License and to reproduce the content of the NOTICE file.
#
#   You may obtain a copy of the Apache License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the Apache License with the above modification is
#   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied. See the Apache License for the specific
#   language governing permissions and limitations under the Apache License.
#

# - Try to find an OpenCL library
# Once done this will define
#
#  OPENCL_FOUND - System has OPENCL
#  OPENCL_INCLUDE_DIRS - The OPENCL include directory
#  OPENCL_LIBRARIES - The libraries needed to use OPENCL


find_package(PackageHandleStandardArgs)

if (APPLE)

    find_library( OPENCL_LIBRARIES
        NAMES
            OpenCL
        DOC "OpenCL lib for OSX"
    )

    find_path( OPENCL_INCLUDE_DIRS
        NAMES
            OpenCL/cl.h
        DOC "Include for OpenCL on OSX"
    )

    find_path( _OPENCL_CPP_INCLUDE_DIRS
        NAMES
            OpenCL/cl.hpp
        DOC "Include for OpenCL CPP bindings on OSX"
    )

elseif (WIN32)

    find_path( OPENCL_INCLUDE_DIRS
         NAMES
             CL/cl.h
    )

    find_path( _OPENCL_CPP_INCLUDE_DIRS
        NAMES
            CL/cl.hpp
    )

    find_library( OPENCL_LIBRARIES
        NAMES
            OpenCL.lib
        PATHS
            ENV OpenCL_LIBPATH
            "$ENV{ATISTREAMSDKROOT}/lib/x86_64"
            "$ENV{ATISTREAMSDKROOT}/lib/x86"
            "C:/Program Files (x86)/Intel/OpenCL SDK/2.0/lib/x64"
            "C:/Program Files (x86)/Intel/OpenCL SDK/2.0/lib/x86"
    )

    get_filename_component( _OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE )

    find_path( OPENCL_INCLUDE_DIRS
        NAMES
            CL/cl.h
        PATHS
            "${_OPENCL_INC_CAND}"
            ENV OpenCL_INCPATH
    )

    find_path( _OPENCL_CPP_INCLUDE_DIRS
        NAMES
            CL/cl.hpp
        PATHS
            "${_OPENCL_INC_CAND}"
            ENV OpenCL_INCPATH
    )

    # AMD APP SDK
    find_file(_OPENCL_DX_INTEROP_CL_D3D11_H
        NAMES
            CL/cl_d3d11.h
        PATHS
            "${_OPENCL_INC_CAND}"
            "${OPENCL_INCLUDE_DIRS}"
            ENV OpenCL_INCPATH
    )
    if (_OPENCL_DX_INTEROP_CL_D3D11_H)
        set(OPENCL_CL_D3D11_H_FOUND "YES")
        message("OpenCL DX11 interop found ${_OPENCL_DX_INTEROP_CL_D3D11_H}")
    endif()

    # NVIDIA GPU Computing ToolKit
    find_file(_OPENCL_DX_INTEROP_CL_D3D11_EXT_H
        NAMES
            CL/cl_d3d11_ext.h
        PATHS
            "${_OPENCL_INC_CAND}"
            "${OPENCL_INCLUDE_DIRS}"
            ENV OpenCL_INCPATH
    )

    if (_OPENCL_DX_INTEROP_CL_D3D11_EXT_H)
        set(OPENCL_CL_D3D11_EXT_H_FOUND "YES")
        message("OpenCL DX11 interop found ${_OPENCL_DX_INTEROP_CL_D3D11_EXT_H}")
    endif()

    if (NOT _OPENCL_DX_INTEROP_CL_D3D11_H AND NOT _OPENCL_DX_INTEROP_CL_D3D11_EXT_H)
        message("OpenCL DX11 interop not found")
    endif()

elseif (UNIX)

    find_library( OPENCL_LIBRARIES
        NAMES
            OpenCL
        PATHS
            ENV LD_LIBRARY_PATH
            ENV OpenCL_LIBPATH
    )

    get_filename_component( OPENCL_LIB_DIR ${OPENCL_LIBRARIES} PATH )

    get_filename_component( _OPENCL_INC_CAND "${OPENCL_LIB_DIR}/../../include" ABSOLUTE )

    find_path( OPENCL_INCLUDE_DIRS
        NAMES
            CL/cl.h
        PATHS
            "${_OPENCL_INC_CAND}"
            "/usr/local/cuda/include"
            "/opt/AMDAPP/include"
            ENV OpenCL_INCPATH
    )

    find_path( _OPENCL_CPP_INCLUDE_DIRS
        NAMES
            CL/cl.hpp
        PATHS
            "${_OPENCL_INC_CAND}"
            "/usr/local/cuda/include"
            "/opt/AMDAPP/include"
            ENV OpenCL_INCPATH
    )

else ()

    message( "Could not determine OpenCL platform" )

endif ()


if(_OPENCL_CPP_INCLUDE_DIRS)

    set( OPENCL_HAS_CPP_BINDINGS TRUE )

    list( APPEND OPENCL_INCLUDE_DIRS ${_OPENCL_CPP_INCLUDE_DIRS} )

    list( REMOVE_DUPLICATES OPENCL_INCLUDE_DIRS )
    
    if(EXISTS "${OPENCL_INCLUDE_DIRS}/CL/cl.h")
    
        file(STRINGS "${OPENCL_INCLUDE_DIRS}/CL/cl.h" LINES REGEX "^#define CL_VERSION_.*$")

        foreach(LINE ${LINES})
        
            string(REGEX MATCHALL "[0-9]+" VERSION ${LINE})
            
            #string(SUBSTRING ${VERSION} 1 2 FOO)
            
            list(GET VERSION 0 MAJOR)
            list(GET VERSION 1 MINOR)
            set(VERSION ${MAJOR}.${MINOR})
            
            if (NOT OPENCL_VERSION OR OPENCL_VERSION VERSION_LESS ${VERSION})
                 set(OPENCL_VERSION ${VERSION})
            endif()

        endforeach()
                   
    endif()

endif(_OPENCL_CPP_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OpenCL 
    REQUIRED_VARS
        OPENCL_LIBRARIES 
        OPENCL_INCLUDE_DIRS
    VERSION_VAR
        OPENCL_VERSION
)

mark_as_advanced( OPENCL_INCLUDE_DIRS )
