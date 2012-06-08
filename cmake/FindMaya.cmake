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

# - Maya finder module
# This module searches for a valid Maya instalation. 
# It searches for Maya's devkit, libraries, executables
# and related paths (scripts)
#
# Variables that will be defined: 
# MAYA_FOUND          Defined if a Maya installation has been detected
# MAYA_EXECUTABLE     Path to Maya's executable
# MAYA_<lib>_FOUND    Defined if <lib> has been found
# MAYA_<lib>_LIBRARY  Path to <lib> library
# MAYA_INCLUDE_DIRS   Path to the devkit's include directories
#
# IMPORTANT: Currently, there's only support for OSX platform and Maya version 2012. 

#=============================================================================
# Copyright 2011-2012 Francisco Requena <frarees@gmail.com>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

SET(MAYA_VERSION_2012 TRUE)

## add one to this list to match your install if none match

IF(APPLE)
  FIND_PATH(MAYA_BASE_DIR include/maya/MFn.h PATH
  	ENV MAYA_LOCATION
  	"/Applications/Autodesk/maya2012.17/Maya.app/Contents"
  	"/Applications/Autodesk/maya2012/Maya.app/Contents"
  	"/Applications/Autodesk/maya2011/Maya.app/Contents"
  	"/Applications/Autodesk/maya2010/Maya.app/Contents"
  	)
  FIND_PATH(MAYA_LIBRARY_DIR libOpenMaya.dylib
    PATHS
      ENV MAYA_LOCATION
      ${MAYA_BASE_DIR}
    PATH_SUFFIXES
	Maya.app/contents/MacOS/
  DOC "Maya's libraries path"
)
ENDIF(APPLE)

IF(UNIX)
  FIND_PATH(MAYA_BASE_DIR include/maya/MFn.h PATH
  	ENV MAYA_LOCATION
  	"/usr/autodesk/maya2012.17-x64"
  	"/usr/autodesk/maya2012-x64"
  	"/usr/autodesk/maya2011-x64"
  	"/usr/autodesk/maya2010-x64"
	)
  FIND_PATH(MAYA_LIBRARY_DIR libOpenMaya.so
    PATHS
      ENV MAYA_LOCATION
      ${MAYA_BASE_DIR}
    PATH_SUFFIXES
	lib/
  DOC "Maya's libraries path"
)
ENDIF(UNIX)
 
IF(WIN32)
  FIND_PATH(MAYA_BASE_DIR include/maya/MFn.h PATH
  	ENV MAYA_LOCATION
	"C:/Program Files/Autodesk/Maya2012-x64"
	"C:/Program Files/Autodesk/Maya2012"
	"C:/Program Files (x86)/Autodesk/Maya2012"
	"C:/Autodesk/maya-2012x64"
	"C:/Program Files/Autodesk/Maya2011-x64"
	"C:/Program Files/Autodesk/Maya2011"
	"C:/Program Files (x86)/Autodesk/Maya2011"
	"C:/Autodesk/maya-2011x64"
	"C:/Program Files/Autodesk/Maya2010-x64"
	"C:/Program Files/Autodesk/Maya2010"
	"C:/Program Files (x86)/Autodesk/Maya2010"
	"C:/Autodesk/maya-2010x64"
	)
  FIND_PATH(MAYA_LIBRARY_DIR OpenMaya.lib
    PATHS
      ENV MAYA_LOCATION
      ${MAYA_BASE_DIR}
    PATH_SUFFIXES
	lib/
  DOC "Maya's libraries path"
)
ENDIF(WIN32)

FIND_PATH(MAYA_INCLUDE_DIR maya/MFn.h
  PATHS
    ENV MAYA_LOCATION
    ${MAYA_BASE_DIR}
  PATH_SUFFIXES
	../../devkit/include/
	include/
  DOC "Maya's devkit headers path"
)

FIND_PATH(MAYA_LIBRARY_DIR OpenMaya
  PATHS
    ENV MAYA_LOCATION
    ${MAYA_BASE_DIR}
  PATH_SUFFIXES
	../../devkit/include/
	include/
  DOC "Maya's devkit headers path"
)

LIST(APPEND MAYA_INCLUDE_DIRS ${MAYA_INCLUDE_DIR})

FIND_PATH(MAYA_DEVKIT_INC_DIR GL/glext.h
  PATHS
    ENV MAYA_LOCATION
    ${MAYA_BASE_DIR}
  PATH_SUFFIXES
	/devkit/plug-ins/
  DOC "Maya's devkit headers path"
)
LIST(APPEND MAYA_INCLUDE_DIRS ${MAYA_DEVKIT_INC_DIR})

FOREACH(MAYA_LIB
  OpenMaya
  OpenMayaAnim
  OpenMayaFX
  OpenMayaRender
  OpenMayaUI  
  Image
  Foundation
  IMFbase
  tbb
  cg
  cgGL
)
  FIND_LIBRARY(MAYA_${MAYA_LIB}_LIBRARY ${MAYA_LIB}
    PATHS
      ENV MAYA_LOCATION
      ${MAYA_BASE_DIR}
    PATH_SUFFIXES
      	MacOS/
	lib/
    DOC "Maya's ${MAYA_LIB} library path"
  )
  
  LIST(APPEND ${MAYA_LIBRARIES} MAYA_${MAYA_LIB}_LIBRARY)
ENDFOREACH(MAYA_LIB)

FIND_PROGRAM(MAYA_EXECUTABLE maya
  PATHS
    ENV MAYA_LOCATION
    ${MAYA_BASE_DIR}
  PATH_SUFFIXES
    	MacOS/
	bin/
  DOC "Maya's executable path"
)

# handle the QUIETLY and REQUIRED arguments and set MAYA_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Maya DEFAULT_MSG ${MAYA_LIBRARIES} MAYA_EXECUTABLE  MAYA_INCLUDE_DIRS)

IF(NOT MAYA_FOUND AND NOT MAYA_FIND_QUIETLY)
  MESSAGE("Remember that currently there's support only for Maya 2012. ")
ENDIF()

