


set(SDKROOT $ENV{SDKROOT})
set(RC_ARCHS $ENV{RC_ARCHS})
separate_arguments(RC_ARCHS)
set(ARCH_FLAGS "-arch arm64")
foreach(ARCH ${RC_ARCHS})
	set(ARCH_FLAGS "${ARCH_FLAGS} -arch ${ARCH}")
endforeach(ARCH)

execute_process(COMMAND xcodebuild -version -sdk ${SDKROOT} PlatformPath OUTPUT_VARIABLE DEVROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND xcodebuild -version -sdk ${SDKROOT} ProductVersion OUTPUT_VARIABLE SDKVER OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND xcodebuild -version -sdk ${SDKROOT} Path OUTPUT_VARIABLE SDKROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND xcrun -sdk ${SDKROOT} -f cc OUTPUT_VARIABLE CC OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND xcrun -sdk ${SDKROOT} -f c++ OUTPUT_VARIABLE CXX OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_SYSTEM_VERSION 9)
set(CMAKE_SYSTEM_PROCESSOR arm)

set (iPhone 1)
set (iPhoneOS 1)
set (iPhoneOS_VERSION ${SDKVER})
set (IOS 1)

set(CMAKE_FIND_ROOT_PATH "${SDKROOT}" "${DEVROOT}")
set(CMAKE_OSX_SYSROOT "${SDKROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)


string(FIND ${DEVROOT} ".platform" PLATFORM_END REVERSE)
string(FIND ${DEVROOT} "/" PLATFORM_START REVERSE)
math(EXPR PLATFORM_START "${PLATFORM_START} + 1")
math(EXPR PLATFORM_LENGTH "${PLATFORM_END} - ${PLATFORM_START}")
string(SUBSTRING ${DEVROOT} ${PLATFORM_START} ${PLATFORM_LENGTH} PLATFORM)
string(TOLOWER ${PLATFORM} PLATFORM)
set(VERSIONMIN "-m${PLATFORM}-version-min=${SDKVER}")

set(DEPLOYMENT_TARGET "${SDKVER}")

set(CMAKE_C_COMPILER "${CC}")
set(CMAKE_CXX_COMPILER "${CXX}")
set(CMAKE_CROSSCOMPILING 1)
set(CMAKE_C_FLAGS "${ARCH_FLAGS} ${VERSIONMIN} -isysroot ${SDKROOT}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${ARCH_FLAGS} ${VERSIONMIN} -isysroot ${SDKROOT}" CACHE STRING "" FORCE)