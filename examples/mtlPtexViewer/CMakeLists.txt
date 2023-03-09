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

# *** mtlPtexViewer ***

if(IOS)
    set(BUNDLE_FILES
        "iOS/Base.lproj/LaunchScreen.storyboard"
        "iOS/Base.lproj/Main.storyboard"
    )
    set(PLATFORM_PLIST "iOS/Info.plist")
    set(PLATFORM_FILES
        "iOS/AppDelegate.h"
        "iOS/AppDelegate.m"
        "iOS/main.m"
        "iOS/ViewController.h"
        "iOS/ViewController.mm"
    )
    list(APPEND PLATFORM_LIBRARIES
        "-framework UIKit"
        "z"
    )
    set_source_files_properties("${BUNDLE_FILES}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
else()
    set(BUNDLE_FILES
        "OSX/MainMenu.xib"
    )
    set(PLATFORM_PLIST "OSX/Info.plist")
    set(PLATFORM_FILES
        "OSX/AppDelegate.h"
        "OSX/AppDelegate.m"
        "OSX/main.m"
        "OSX/ViewController.h"
        "OSX/ViewController.mm"
    )
    list(APPEND PLATFORM_LIBRARIES
        "-framework OpenGL"
        "-framework AppKit"
    )
    set_source_files_properties("${BUNDLE_FILES}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
endif()

include_directories(
    "${OPENSUBDIV_INCLUDE_DIR}"
    "${METAL_INCLUDE_DIR}"
    "${PTEX_INCLUDE_DIR}"
)

list(APPEND PLATFORM_FILES
    "mtlPtexViewer.h"
    "mtlPtexViewer.mm"
)

list(APPEND PLATFORM_LIBRARIES
    "osd_static_framework"
    "${METAL_LIBRARY}"
    "-framework MetalKit"
    "-framework Foundation"
    "-framework QuartzCore"
    "${PTEX_LIBRARY}"
)

osd_stringify("mtlPtexViewer.metal" INC_FILES)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

osd_add_executable(mtlPtexViewer "examples/mtlPtexViewer"
    MACOSX_BUNDLE
    "${PLATFORM_FILES}"
    "${INC_FILES}"
    "${BUNDLE_FILES}"
    $<TARGET_OBJECTS:regression_common_obj>
    $<TARGET_OBJECTS:regression_far_utils_obj>
    $<TARGET_OBJECTS:examples_common_mtl_obj>
    $<TARGET_OBJECTS:examples_common_ptex_obj>
)


set_target_properties (mtlPtexViewer PROPERTIES 
    LINKER_LANGUAGE CXX
    RESOURCE "${BUNDLE_FILES}"
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_LIST_DIR}/${PLATFORM_PLIST}"
)

if(IOS)
    set_target_properties(mtlPtexViewer PROPERTIES
        XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
    )
endif()

target_link_libraries(mtlPtexViewer
    ${PLATFORM_LIBRARIES}
)


install(TARGETS mtlPtexViewer DESTINATION "${CMAKE_BINDIR_BASE}")
