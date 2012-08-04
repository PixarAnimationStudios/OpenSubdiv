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

