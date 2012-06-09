find_package(IlmBase REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

include_directories(
    ${ILMBASE_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/opensubdiv
)


set(SOURCE_FILES
    cpuDispatcher.cpp
    cpuKernel.cpp
    kernelDispatcher.cpp
    mesh.cpp
)

set(HEADER_FILES
    cpuDispatcher.h
    cpuKernel.h
    kernelDispatcher.h
    local.h
    mesh.h
    vertex.h
)

if(APPLE)
    set(PLATFORM_COMPILE_FLAGS
	-fopenmp
    )
    set(PLATFORM_LIBRARIES
        #XXX:cleanup  Remove opengl, glut, and OpenCL from this list.
	/System/Library/Frameworks/OpenGL.framework
	/System/Library/Frameworks/OpenCL.framework
	/System/Library/Frameworks/GLUT.framework
	gomp
    )
endif(APPLE)

if(UNIX AND NOT APPLE)
    set(PLATFORM_COMPILE_FLAGS
        -fopenmp
        -fPIC
    )
    set(PLATFORM_LIBRARIES
        GLU
	glut
        gomp
    )
endif(UNIX AND NOT APPLE)

if(WIN32)
    set(PLATFORM_COMPILE_FLAGS
        /openmp
    )
    set(PLATFORM_LIBRARIES
    )
endif(WIN32)

macro(_add_library target)
    if(CUDA_FOUND)
        cuda_add_library(${target} ${ARGN})
    else()
        add_library(${target} ${ARGN})
    endif()
endmacro()

_add_library(osd STATIC
    ${SOURCE_FILES}
    ${HEADER_FILES}
)

target_link_libraries(osd
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${PLATFORM_LIBRARIES}
    glew
)
