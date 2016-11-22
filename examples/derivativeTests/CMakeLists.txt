
include_directories(
    "${OPENSUBDIV_INCLUDE_DIR}"
)

list(APPEND PLATFORM_LIBRARIES
    "${OSD_LINK_TARGET}"
)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

_add_executable(derivativeTests "examples"
    derivativeTests.cpp
)

target_link_libraries(derivativeTests
    ${PLATFORM_LIBRARIES}
)

install(TARGETS derivativeTests DESTINATION "${CMAKE_BINDIR_BASE}")
