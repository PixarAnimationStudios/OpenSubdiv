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

# - Try to find OpenGLES
# Once done this will define
#  
#  OPENGLES_FOUND        - system has OpenGLES
#  OPENGLES_INCLUDE_DIR  - the GL include directory
#  OPENGLES_LIBRARIES    - Link these to use OpenGLES

if(ANDROID)
    FIND_PATH(
            OPENGLES_INCLUDE_DIR GLES2/gl2.h 
            ${ANDROID_STANDALONE_TOOLCHAIN}/usr/include
    )

    FIND_LIBRARY(
            OPENGLES_LIBRARIES NAMES  GLESv2
            PATHS ${ANDROID_STANDALONE_TOOLCHAIN}/usr/lib
    )

elseif(IOS)
    FIND_LIBRARY(
            OPENGLES_FRAMEWORKS OpenGLES
    )
    if(OPENGLES_FRAMEWORKS)
        set( OPENGLES_LIBRARIES "-framework OpenGLES" )
    endif()

endif()

SET( OPENGLES_FOUND "NO" )
IF(OPENGLES_LIBRARIES)
    SET( OPENGLES_FOUND "YES" )
ENDIF(OPENGLES_LIBRARIES)

MARK_AS_ADVANCED(
  OPENGLES_INCLUDE_DIR
  OPENGLES_LIBRARIES
)

