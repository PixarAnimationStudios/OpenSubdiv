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

# - Try to find the IlmBase library
# Once done this will define
#
#  ILMBASE_FOUND - System has IlmBase
#  ILMBASE_INCLUDE_DIR - The include directory
#  ILMBASE_LIBRARIES - The libraries needed

IF(NOT DEFINED ILMBASE_LOCATION)
    IF ( ${CMAKE_HOST_UNIX} )
        IF( APPLE )
          # TODO: set to default install path when shipping out
          SET( ILMBASE_LOCATION NOTFOUND )
        ELSE()
          SET(ILMBASE_LOCATION /usr/local/ilmbase-1.0.1 )
        ENDIF()
    ELSE()
        IF ( WIN32 )
          # Note: This assumes that the Deploy directory has been copied
          #       back into the IlmBase root directory.
          SET( ILMBASE_LOCATION $ENV{PROGRAMFILES}/ilmbase-1.0.1/Deploy )
        ENDIF()
    ENDIF()
ENDIF()

IF ( ${CMAKE_HOST_UNIX} )
    SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -lpthread")
ELSE()
ENDIF()

SET(LIBRARY_PATHS
    ${ILMBASE_LOCATION}/lib
    ${ILMBASE_LOCATION}/lib/Release
    ${ILMBASE_LOCATION}/lib/x64/Release
    $ENV{ILMBASE_LOCATION}/lib
    $ENV{ILMBASE_LOCATION}/lib/Release
    $ENV{ILMBASE_LOCATION}/lib/x64/Release
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local/lib
    /usr/lib
    /sw/lib
    /opt/local/lib
    /opt/csw/lib
    /opt/lib
    /usr/freeware/lib64
)

SET(INCLUDE_PATHS
    ${ILMBASE_LOCATION}/include/OpenEXR/
    ${ILMBASE_LOCATION}/include
    $ENV{ILMBASE_LOCATION}/include/OpenEXR/
    $ENV{ILMBASE_LOCATION}/include
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local/include/OpenEXR/
    /usr/local/include
    /usr/include
    /usr/include/OpenEXR
    /sw/include # Fink
    /opt/local/include # DarwinPorts
    /opt/csw/include # Blastwave
    /opt/include
    /usr/freeware/include
)


FIND_PATH( ILMBASE_INCLUDE_DIR ImathMath.h
           PATHS
           ${INCLUDE_PATHS}
           NO_DEFAULT_PATH
           NO_CMAKE_ENVIRONMENT_PATH
           NO_CMAKE_PATH
           NO_SYSTEM_ENVIRONMENT_PATH
           NO_CMAKE_SYSTEM_PATH
           DOC "The directory where ImathMath.h resides" )

FIND_LIBRARY( ILMBASE_IEX_LIB Iex
              PATHS
              ${LIBRARY_PATHS}
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH
              DOC "The Iex library" )

FIND_LIBRARY( ILMBASE_ILMTHREAD_LIB IlmThread
              PATHS
              ${LIBRARY_PATHS}
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH
              DOC "The IlmThread library" )

FIND_LIBRARY( ILMBASE_IMATH_LIB Imath
              PATHS
              ${LIBRARY_PATHS}
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH
              DOC "The Imath library" )


IF ( ${ILMBASE_IEX_LIB} STREQUAL "ILMBASE_IEX_LIB-NOTFOUND" )
  MESSAGE( FATAL_ERROR "ilmbase libraries (Iex) not found, required: ILMBASE_LOCATION: ${ILMBASE_LOCATION}" )
ENDIF()

IF ( ${ILMBASE_ILMTHREAD_LIB} STREQUAL "ILMBASE_ILMTHREAD_LIB-NOTFOUND" )
  MESSAGE( FATAL_ERROR "ilmbase libraries (IlmThread) not found, required: ILMBASE_LOCATION: ${ILMBASE_LOCATION}" )
ENDIF()

IF ( ${ILMBASE_IMATH_LIB} STREQUAL "ILMBASE_IMATH_LIB-NOTFOUND" )
  MESSAGE( FATAL_ERROR "ilmbase libraries (Imath) not found, required: ILMBASE_LOCATION: ${ILMBASE_LOCATION}" )
ENDIF()

IF ( ${ILMBASE_INCLUDE_DIR} STREQUAL "ILMBASE_INCLUDE_DIR-NOTFOUND" )
  MESSAGE( FATAL_ERROR "ilmbase header files not found, required: ILMBASE_LOCATION: ${ILMBASE_LOCATION}" )
ENDIF()

SET( ILMBASE_LIBRARIES
       ${ILMBASE_IMATH_LIB}
       ${ILMBASE_ILMTHREAD_LIB}
       ${ILMBASE_IEX_LIB}
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(IlmBase DEFAULT_MSG
    ILMBASE_LOCATION
    ILMBASE_INCLUDE_DIR
    ILMBASE_LIBRARIES
)

SET( ILMBASE_FOUND TRUE )

