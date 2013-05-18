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

# - Try to find Intel's Threading Building Blocks
# Once done this will define
#
#  TBB_FOUND - System has OPENCL
#  TBB_INCLUDE_DIR - The TBB include directory
#  TBB_LIBRARIES - The libraries needed to use TBB

# Obtain include directory
if (WIN32)
    find_path(TBB_INCLUDE_DIR
        NAMES
            tbb/tbb.h
        PATHS
            ${TBB_LOCATION}/include
            $ENV{TBB_LOCATION}/include
            $ENV{PROGRAMFILES}/Intel/TBB/include
            /usr/include
            DOC "The directory where TBB headers reside")
elseif (APPLE)
    find_path(TBB_INCLUDE_DIR
        NAMES
            tbb/tbb.h
        PATHS
            ${TBB_LOCATION}/include
            $ENV{TBB_LOCATION}/include
            DOC "The directory where TBB headers reside")
else ()
    find_path(TBB_INCLUDE_DIR
        NAMES
            tbb/tbb.h
        PATHS
            ${TBB_LOCATION}/include
            $ENV{TBB_LOCATION}/include
            /usr/include
            /usr/local/include
            /usr/openwin/share/include
            /usr/openwin/include
            DOC "The directory where TBB headers reside")
endif ()

# List library files
foreach(TBB_LIB tbb             tbb_debug 
                tbbmalloc       tbbmalloc_debug
                tbbmalloc_proxy tbbmalloc_proxy_debug
                tbb_preview     tbb_preview_debug)

    if (WIN32)

            if ("${CMAKE_GENERATOR}" MATCHES "[Ww]in64")
            set(WINPATH intel64)
        else ()
            set(WINPATH ia32)
        endif()

            if (MSVC80)
            set(WINPATH "${WINPATH}/vc8")
        elseif (MSVC90)
            set(WINPATH "${WINPATH}/vc9")
        elseif (MSVC10)
            set(WINPATH "${WINPATH}/vc10")
        elseif (MSVC11)
            set(WINPATH "${WINPATH}/vc11")
        endif()
    endif()

    find_library(TBB_${TBB_LIB}_LIBRARY
        NAMES
            ${TBB_LIB}
        PATHS
            ${TBB_LOCATION}/lib
            ${TBB_LOCATION}/bin/${WINPATH}
            ${TBB_LOCATION}/lib/${WINPATH}
            $ENV{TBB_LOCATION}/lib
            $ENV{TBB_LOCATION}/bin/${WINPATH}
            $ENV{PROGRAMFILES}/TBB/lib
            /usr/lib
            /usr/lib/w32api
            /usr/local/lib
            /usr/X11R6/lib
            DOC "Intel's Threading Building Blocks library")

    if (TBB_${TBB_LIB}_LIBRARY)
        list(APPEND TBB_LIBRARIES ${TBB_${TBB_LIB}_LIBRARY})
    endif()
    
endforeach()

# Obtain version information
if(TBB_INCLUDE_DIR)

    # Tease the TBB version numbers from the lib headers
    function(parseVersion FILENAME VARNAME)
            
        set(PATTERN "^#define ${VARNAME}.*$")
        
        file(STRINGS "${TBB_INCLUDE_DIR}/${FILENAME}" TMP REGEX ${PATTERN})
        
        string(REGEX MATCHALL "[0-9]+" TMP ${TMP})
        
        set(${VARNAME} ${TMP} PARENT_SCOPE)
        
    endfunction()

    if(EXISTS "${TBB_INCLUDE_DIR}/tbb/tbb_stddef.h")
        parseVersion(tbb/tbb_stddef.h TBB_VERSION_MAJOR)
        parseVersion(tbb/tbb_stddef.h TBB_VERSION_MINOR)        
    endif()

    if(${TBB_VERSION_MAJOR} OR ${TBB_VERSION_MINOR})
        set(TBB_VERSION "${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}")
        set(TBB_VERSION_STRING "${TBB_VERSION}")
        mark_as_advanced(TBB_VERSION)
    endif()

endif(TBB_INCLUDE_DIR)




include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(TBB 
    REQUIRED_VARS
        TBB_INCLUDE_DIR
        TBB_LIBRARIES
    VERSION_VAR
        TBB_VERSION
)

mark_as_advanced(
  TBB_INCLUDE_DIR
  TBB_LIBRARIES
)

