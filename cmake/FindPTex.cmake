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

# Try to find PTex library and include path.
# Once done this will define
#
# PTEX_FOUND
# PTEX_INCLUDE_DIR
# PTEX_LIBRARY
#

if (WIN32)
    find_path( PTEX_INCLUDE_DIR
        NAMES
            Ptexture.h
        PATHS
            ${PTEX_LOCATION}/include
            $ENV{PTEX_LOCATION}/include
            $ENV{PROGRAMFILES}/Ptex/include
            /usr/include
            DOC "The directory where Ptexture.h resides")
    find_library( PTEX_LIBRARY
        NAMES
            Ptex32 Ptex32s Ptex
        PATHS
            ${PTEX_LOCATION}/lib
            $ENV{PTEX_LOCATION}/lib
            $ENV{PROGRAMFILES}/Ptex/lib
            /usr/lib
            /usr/lib/w32api
            /usr/local/lib
            /usr/X11R6/lib
            DOC "The Ptex library")
elseif (APPLE)
    find_path( PTEX_INCLUDE_DIR
        NAMES
            Ptexture.h
        PATHS
            ${PTEX_LOCATION}/include
            $ENV{PTEX_LOCATION}/include
            DOC "The directory where Ptexture.h resides")
    find_library( PTEX_LIBRARY
        NAMES
            libPtex.a
        PATHS
            ${PTEX_LOCATION}/lib
            $ENV{PTEX_LOCATION}/lib
            DOC "The Ptex Library")
else ()
    find_path( PTEX_INCLUDE_DIR
        NAMES
            Ptexture.h
        PATHS
            ${PTEX_LOCATION}/include
            ${PTEX_LOCATION}/include/wdas
            $ENV{PTEX_LOCATION}/include
            $ENV{PTEX_LOCATION}/include/wdas
            /usr/include
            /usr/local/include
            /usr/openwin/share/include
            /usr/openwin/include
            /usr/X11R6/include
            /usr/include/X11
            DOC "The directory where Ptexture.h resides")
    find_library( PTEX_LIBRARY
        NAMES
            Ptex wdasPtex
        PATHS
            ${PTEX_LOCATION}/lib
            $ENV{PTEX_LOCATION}/lib
            /usr/lib
            /usr/local/lib
            /usr/openwin/lib
            /usr/X11R6/lib
            DOC "The Ptex library")
endif ()

if (PTEX_INCLUDE_DIR AND EXISTS "${PTEX_INCLUDE_DIR}/Ptexture.h" )

    file(STRINGS "${PTEX_INCLUDE_DIR}/Ptexture.h" TMP REGEX "^#define PtexAPIVersion.*$")
    string(REGEX MATCHALL "[0-9]+" API ${TMP})
    
    file(STRINGS "${PTEX_INCLUDE_DIR}/Ptexture.h" TMP REGEX "^#define PtexFileMajorVersion.*$")
    string(REGEX MATCHALL "[0-9]+" MAJOR ${TMP})

    file(STRINGS "${PTEX_INCLUDE_DIR}/Ptexture.h" TMP REGEX "^#define PtexFileMinorVersion.*$")
    string(REGEX MATCHALL "[0-9]+" MINOR ${TMP})

    set(PTEX_VERSION ${API}.${MAJOR}.${MINOR})

endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(PTex 
    REQUIRED_VARS
        PTEX_INCLUDE_DIR
        PTEX_LIBRARY
    VERSION_VAR
        PTEX_VERSION
)

mark_as_advanced(
  PTEX_INCLUDE_DIR
  PTEX_LIBRARY
)

