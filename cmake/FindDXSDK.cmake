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
            "${DXSDK_LOCATION}/Include"
            "$ENV{DXSDK_LOCATION}/Include"
            "${DXSDK_ROOT}/Include"
            "$ENV{DXSDK_ROOT}/Include"
            "C:/Program Files (x86)/Windows Kits/10/Include/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/um"
            "C:/Program Files (x86)/Windows Kits/8.1/Include/um"
            "C:/Program Files (x86)/Microsoft DirectX SDK*/Include"
            "C:/Program Files/Microsoft DirectX SDK*/Include"
    )

    if ("${CMAKE_GENERATOR}" MATCHES "[Ww]in64" OR
        "${CMAKE_GENERATOR_PLATFORM}" MATCHES "x64")
        set(ARCH x64)
    else()
        set(ARCH x86)
    endif()

    find_path(LIBRARY_DIR
            d3d11.lib
        PATHS
            "${DXSDK_LOCATION}/Lib/${ARCH}"
            "$ENV{DXSDK_LOCATION}/Lib/${ARCH}"
            "${DXSDK_ROOT}/Lib/${ARCH}"
            "$ENV{DXSDK_ROOT}/Lib/${ARCH}"
            "C:/Program Files (x86)/Windows Kits/10/Lib/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/um/${ARCH}"
            "C:/Program Files (x86)/Windows Kits/8.1/Lib/winv6.3/um/${ARCH}"
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
            NO_DEFAULT_PATH
        )

        if(NOT DXSDK_${DX_LIB}_LIBRARY)
            message(FATAL_ERROR "Could not find required library ${DX_LIB}")
        else()
            list(APPEND DXSDK_LIBRARIES ${DXSDK_${DX_LIB}_LIBRARY})
        endif()


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

