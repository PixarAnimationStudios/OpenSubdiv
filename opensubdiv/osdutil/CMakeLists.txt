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

#-------------------------------------------------------------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/opensubdiv
)

#-------------------------------------------------------------------------------
# source & headers

set(PUBLIC_HEADER_FILES
    batch.h
    drawItem.h
    drawController.h
)

set(DOXY_HEADER_FILES ${PUBLIC_HEADER_FILES})

#-------------------------------------------------------------------------------
# OpenCL code & dependencies

set(OPENCL_PUBLIC_HEADER_FILES
    batchCL.h
)

if ( OPENCL_FOUND )
    list(APPEND PUBLIC_HEADER_FILES ${OPENCL_PUBLIC_HEADER_FILES} )
endif()

list(APPEND DOXY_HEADER_FILES ${OPENCL_PUBLIC_HEADER_FILES} )

#-------------------------------------------------------------------------------

_add_doxy_headers( "${DOXY_HEADER_FILES}" )

install( FILES ${PUBLIC_HEADER_FILES}
         DESTINATION ${CMAKE_INCDIR_BASE}/osdutil
         PERMISSIONS OWNER_READ GROUP_READ WORLD_READ )

#-------------------------------------------------------------------------------
