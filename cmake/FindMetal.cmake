if(APPLE)
    find_path( METAL_INCLUDE_DIR
        Metal/Metal.h
    )

    find_library( METAL_FRAMEWORKS Metal )

    if(METAL_FRAMEWORKS)
        set( METAL_LIBRARIES "-framework Metal -framework Foundation" )
    endif()
endif()


set( METAL_FOUND "NO" )
if(METAL_LIBRARIES)
    set( METAL_FOUND "YES" )
endif(METAL_LIBRARIES)

mark_as_advanced(
  METAL_INCLUDE_DIR
  METAL_LIBRARIES
)
