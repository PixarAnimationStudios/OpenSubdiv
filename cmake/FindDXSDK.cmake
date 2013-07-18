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

# Try to find DirectX SDK.
# Once done this will define
#
# DXSDK_FOUND
# DXSDK_INCLUDE_DIR
# DXSDK_LIBRARY_DIR
# DXSDK_LIBRARIES
# DXSDK_LOCATION
#
# Also will define

if (WIN32)
    find_path(DXSDK_INCLUDE_DIR
        NAMES
            D3D11.h D3Dcompiler.h
        PATHS
            ${DXSDK_LOCATION}/Include
            $ENV{DXSDK_LOCATION}/Include
            ${DXSDK_ROOT}/Include
            $ENV{DXSDK_ROOT}/Include
            "C:/Program Files (x86)/Microsoft DirectX SDK*/Include"
            "C:/Program Files/Microsoft DirectX SDK*/Include"
    )

    if ("${CMAKE_GENERATOR}" MATCHES "[Ww]in64")
        set(ARCH x64)
    else()
        set(ARCH x86)
    endif()

    find_path(LIBRARY_DIR
            d3d11.lib
        PATHS
            ${DXSDK_LOCATION}/Lib/${ARCH}
            $ENV{DXSDK_LOCATION}/Lib/${ARCH}
            ${DXSDK_ROOT}/Lib/${ARCH}
            $ENV{DXSDK_ROOT}/Lib/${ARCH}
            "C:/Program Files (x86)/Microsoft DirectX SDK*/Lib/${ARCH}"
            "C:/Program Files/Microsoft DirectX SDK*/Lib/${ARCH}"
    )

    set(DXSDK_LIBRARY_DIR ${LIBRARY_DIR})

    foreach(DX_LIB d3d11 d3dcompiler)

        find_library(DXSDK_${DX_LIB}_LIBRARY
            NAMES 
                ${DX_LIB}.lib
            PATHS
                ${DXSDK_LIBRARY_DIR}
        )

        list(APPEND DXSDK_LIBRARIES ${DXSDK_${DX_LIB}_LIBRARY})


    endforeach(DX_LIB)

endif ()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(DXSDK DEFAULT_MSG
    DXSDK_INCLUDE_DIR
    DXSDK_LIBRARY_DIR
    DXSDK_LIBRARIES
)

mark_as_advanced(
    DXSDK_INCLUDE_DIR
    DXSDK_LIBRARY_DIR
    DXSDK_LIBRARIES
)

