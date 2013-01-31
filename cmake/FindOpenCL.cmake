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

# - Try to find an OpenCL library
# Once done this will define
#
#  OPENCL_FOUND - System has OPENCL
#  OPENCL_INCLUDE_DIR - The OPENCL include directory
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

elseif (UNIX)

    find_library( OPENCL_LIBRARIES
        NAMES
            OpenCL
        PATHS
            ENV LD_LIBRARY_PATH
            ENV OpenCL_LIBPATH
    )

    get_filename_component( OPENCL_LIB_DIR ${OPENCL_LIBRARIES} PATH )

    get_filename_component( _OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE )

    find_path( OPENCL_INCLUDE_DIRS
        NAMES
            CL/cl.h
        PATHS
            ${_OPENCL_INC_CAND}
            "/usr/local/cuda/include"
            "/opt/AMDAPP/include"
            ENV OpenCL_INCPATH
    )

    find_path( _OPENCL_CPP_INCLUDE_DIRS
        NAMES
            CL/cl.hpp
        PATHS
            ${_OPENCL_INC_CAND}
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
