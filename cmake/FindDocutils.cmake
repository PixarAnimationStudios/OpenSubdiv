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

# - Try to find the restructured text to HTML converter
#
# Once done this will define
#
#  DOCUTILS_FOUND - System has Docutils
#
#  RST2HTML_EXECUTALE

find_package(PackageHandleStandardArgs)

find_program( RST2HTML_EXECUTABLE 
    NAMES 
        rst2html.py 
        rst2html
    DOC 
        "The Python Docutils reStructuredText HTML converter"
)

if (RST2HTML_EXECUTABLE)

    set(DOCUTILS_FOUND "YES")

    # find the version
    execute_process( COMMAND ${RST2HTML_EXECUTABLE} --version OUTPUT_VARIABLE VERSION_STRING )

    # ex : rst2html (Docutils 0.6 [release], Python 2.6.6, on linux2)
    string(REGEX MATCHALL "([^\ ]+\ |[^\ ]+$)" VERSION_TOKENS "${VERSION_STRING}")  
    
    # isolate the 3rd. word in the string
    list (GET VERSION_TOKENS 2 VERSION_STRING)
    
    # remove white space
    string(REGEX REPLACE "[ \t]+$" "" VERSION_STRING ${VERSION_STRING})
    string(REGEX REPLACE "^[ \t]+" "" VERSION_STRING ${VERSION_STRING})
    
    set(DOCUTILS_VERSION ${VERSION_STRING})
        
else()

    set(DOCUTILS_FOUND "NO")

endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Docutils 
    REQUIRED_VARS
        RST2HTML_EXECUTABLE
    VERSION_VAR
        DOCUTILS_VERSION
)

mark_as_advanced(
    RST2HTML_EXECUTABLE
)
