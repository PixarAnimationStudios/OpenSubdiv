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

set(MAYA_FIND_QUIETLY TRUE)

find_package(Maya)

if(NOT MAYA_FOUND)
    message(WARNING
        "Maya could not be found, so the OpenSubdiv Maya plugin will not "
        "be available.  If you do have Maya installed and see this message, "
        "please add your Maya path to cmake/FindMaya.cmake or set it in "
        "the MAYA_LOCATION environment variable."
    )
    return()
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/opensubdiv
    ${MAYA_INCLUDE_DIRS}
    ${ILMBASE_INCLUDE_DIR}
)

set(SOURCE_FILES
    OpenSubdivDrawOverride.cpp
    hbrUtil.cpp
)

set(HEADER_FILES
)

if(UNIX)
    set(PLATFORM_COMPILE_FLAGS
        -D_BOOL
        -DREQUIRE_IOSTREAM
    )
    set(PLATFORM_LIBRARIES
    )
    set(PLATFORM_PLUGIN_EXTENSION
        .so
    )
    set(PLATFORM_LINK_FLAGS
    )
endif(UNIX)

if(WIN32)
    set(PLATFORM_COMPILE_FLAGS
        /D_AFXDLL
        /DNT_PLUGIN
        /DREQUIRE_IOSTREAM
    )
    set(PLATFORM_LIBRARIES
    )
    set(PLATFORM_PLUGIN_EXTENSION
        .mll
    )
    set(PLATFORM_LINK_FLAGS
        "/export:initializePlugin /export:uninitializePlugin"
    )
endif(WIN32)

add_definitions(
    ${PLATFORM_COMPILE_FLAGS}
)

add_library(maya_plugin SHARED
    ${SOURCE_FILES}
    ${HEADER_FILES}
)

set_target_properties(maya_plugin
    PROPERTIES
    OUTPUT_NAME "mayaViewer"
    SUFFIX ${PLATFORM_PLUGIN_EXTENSION}
    LINK_FLAGS "${PLATFORM_LINK_FLAGS}"
)

target_link_libraries(maya_plugin
    osd
    ${MAYA_Foundation_LIBRARY}
    ${MAYA_OpenMaya_LIBRARY}
    ${MAYA_OpenMayaRender_LIBRARY}
    ${MAYA_tbb_LIBRARY}
    ${PLATFORM_LIBRARIES}
    GLEW
)
