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
#

# Try to find GLUT library and include path.
# Once done this will define
#
# GLUT_FOUND
# GLUT_INCLUDE_DIR
# GLUT_LIBRARIES
# 

IF (WIN32)

  IF(CYGWIN)

    FIND_PATH( GLUT_INCLUDE_DIR GL/glut.h
      ${GLUT_LOCATION}/include
      /usr/include
    )

    FIND_LIBRARY( GLUT_glut_LIBRARY glut32
      ${GLUT_LOCATION}/lib
      ${OPENGL_LIBRARY_DIR}
      /usr/lib
      /usr/lib/w32api
      /usr/local/lib
      /usr/X11R6/lib
    )


  ELSE(CYGWIN)

	FIND_PATH( GLUT_INCLUDE_DIR GL/glut.h
		${GLUT_LOCATION}/include
		${PROJECT_SOURCE_DIR}/extern/glut/include
		DOC "The directory where GL/glut.h resides")
	FIND_LIBRARY( GLUT_glut_LIBRARY
		NAMES glut GLUT glut32 glut32s
		PATHS
		${GLUT_LOCATION}/lib
		${PROJECT_SOURCE_DIR}/extern/glut/bin
		${PROJECT_SOURCE_DIR}/extern/glut/lib
		${OPENGL_LIBRARY_DIR}
		DOC "The GLUT library")

  ENDIF(CYGWIN)
ELSE (WIN32)

  IF (APPLE)
# These values for Apple could probably do with improvement.
    FIND_PATH( GLUT_INCLUDE_DIR glut.h
      /System/Library/Frameworks/GLUT.framework/Versions/A/Headers
      ${OPENGL_LIBRARY_DIR}
    )
    SET(GLUT_glut_LIBRARY "-framework Glut" CACHE STRING "GLUT library for OSX") 
    SET(GLUT_cocoa_LIBRARY "-framework Cocoa" CACHE STRING "Cocoa framework for OSX")
  ELSE (APPLE)

    FIND_PATH( GLUT_INCLUDE_DIR GL/glut.h
      ${GLUT_LOCATION}/include
      /usr/include
      /usr/include/GL
      /usr/local/include
      /usr/openwin/share/include
      /usr/openwin/include
      /usr/X11R6/include
      /usr/include/X11
      /opt/graphics/OpenGL/include
      /opt/graphics/OpenGL/contrib/libglut
    )

    FIND_LIBRARY( GLUT_glut_LIBRARY glut
      ${GLUT_LOCATION}/lib
      /usr/lib
      /usr/local/lib
      /usr/openwin/lib
      /usr/X11R6/lib
    )

    FIND_LIBRARY( GLUT_Xi_LIBRARY Xi
      ${GLUT_LOCATION}/lib
      /usr/lib
      /usr/local/lib
      /usr/openwin/lib
      /usr/X11R6/lib
    )

    FIND_LIBRARY( GLUT_Xmu_LIBRARY Xmu
      ${GLUT_LOCATION}/lib
      /usr/lib
      /usr/local/lib
      /usr/openwin/lib
      /usr/X11R6/lib
    )

  ENDIF (APPLE)

ENDIF (WIN32)

SET( GLUT_FOUND "NO" )
IF(GLUT_INCLUDE_DIR)
  IF(GLUT_glut_LIBRARY)
    # Is -lXi and -lXmu required on all platforms that have it?
    # If not, we need some way to figure out what platform we are on.
    SET( GLUT_LIBRARIES
      ${GLUT_glut_LIBRARY}
      ${GLUT_Xmu_LIBRARY}
      ${GLUT_Xi_LIBRARY} 
      ${GLUT_cocoa_LIBRARY}
    )
    SET( GLUT_FOUND "YES" )

    SET (GLUT_LIBRARY ${GLUT_LIBRARIES})
    SET (GLUT_INCLUDE_PATH ${GLUT_INCLUDE_DIR})

  ENDIF(GLUT_glut_LIBRARY)
ENDIF(GLUT_INCLUDE_DIR)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GLUT DEFAULT_MSG
    GLUT_INCLUDE_DIR
    GLUT_LIBRARIES
)


MARK_AS_ADVANCED(
  GLUT_INCLUDE_DIR
  GLUT_glut_LIBRARY
  GLUT_Xmu_LIBRARY
  GLUT_Xi_LIBRARY
)
  
