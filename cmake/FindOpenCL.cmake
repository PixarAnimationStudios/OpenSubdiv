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

# - Try to find an OpenCL library
# Once done this will define
#
#  OPENCL_FOUND - System has OPENCL
#  OPENCL_INCLUDE_DIR - The OPENCL include directory
#  OPENCL_LIBRARIES - The libraries needed to use OPENCL
#

# Copyright (c) 2010 Matthieu Volat. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing
# official policies, either expressed or implied, of Matthieu Volat.

IF(OPENCL_LIBRARIES)
   SET(OpenCL_FIND_QUIETLY TRUE)
ENDIF(OPENCL_LIBRARIES)

IF(APPLE)
  SET(CL_DIR OpenCL)
ELSE(APPLE)
  SET(CL_DIR CL)
ENDIF(APPLE)
FIND_PATH(OPENCL_INCLUDE_DIR ${CL_DIR}/opencl.h 
    PATHS 
        /usr/local/cuda/include
        $ENV{AMDAPPSDKROOT}/include
        /System/Library/Frameworks/OpenCL.framework/Headers
    PATH_SUFFIXES nvidia)

IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
  SET(AMD_ARCH_LIBDIR x86_64)
  SET(WIN_LIBDIR Win64)
ELSE(CMAKE_SIZEOF_VOID_P EQUAL 8)
  SET(AMD_ARCH_LIBDIR x86)
  SET(WIN_LIBDIR Win32)
ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 8)
FIND_LIBRARY(OPENCL_LIBRARIES 
    NAMES OpenCL 
    PATH
        $ENV{AMDAPPSDKROOT}/lib/${AMD_ARCH_LIBDIR}
    PATH_SUFFIXES nvidia nvidia-current ${WIN_LIBDIR})

IF(OPENCL_INCLUDE_DIR AND OPENCL_LIBRARIES)
   SET(OPENCL_FOUND TRUE)
ELSE(OPENCL_INCLUDE_DIR AND OPENCL_LIBRARIES)
   SET(OPENCL_FOUND FALSE)
ENDIF (OPENCL_INCLUDE_DIR AND OPENCL_LIBRARIES)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCL DEFAULT_MSG
  OPENCL_LIBRARIES 
  OPENCL_INCLUDE_DIR
)

MARK_AS_ADVANCED(OPENCL_INCLUDE_DIR OPENCL_LIBRARIES)

