..
     Copyright 2026 Pixar

     Licensed under the terms set forth in the LICENSE.txt file available at
     https://opensubdiv.org/license.


Overview of Release 3.8
=======================

.. contents::
   :local:
   :backlinks: none

Updates
-------

This release contains API additions along with updates to improve
configuration and compatibilty when working with recent compilers,
operating systems, and other software.

The OpenSubdiv::Bfr interface has been updated with minor API additions
and documentation improvements based on feedback from production users.

Significant updates include:
 - C++17 is set as the default C++ standard.
 - Additional compiler warnings have been addressed when building with C++20.
 - CMake minimum required and policy max versions are set to 3.17 and 3.21 respectively.
 - CMake's FindCUDAToolkit is now used instead of FindCUDA when CUDA support is enabled.
 - CMAKE_INSTALL_RPATH_USE_LINK_PATH is no longer set by default but instead can be enabled as needed during CMake configuration.

Other Changes
-------------

Improvements
~~~~~~~~~~~~
    - Updated C++ Standard to C++17 (GitHub #1388)
    - Updated CMake minimum required and policy max versions to 3.17 and 3.21 respectively (GitHub #1393)
    - Updated to use FindCUDAToolkit (GitHub #1395)
    - Updated default rpath link behavior (GitHub #1376, #1402)
    - Updated public constants with inline use of constexpr (GitHub #1392)
    - Added alternative constructor for Bfr::Parameterization (GitHub #1391)
    - Clarified use of Bfr::Parameterization::Type in documentation (GitHub #1397)
    - Improved Bfr::Tessellation quad-preserving patterns (GitHub #1386)
    - Added const qualifier to Bfr::Tessellation transform methods (GitHub #1385)
    - Addressed compiler warnings (GitHub #1387, #1389)
    - Improved processing of compile and link flags (GitHub #1390, #1394)
    - Updated recommended package dependencies (GitHub #1398)
    - Addressed minor documentation issues (GitHub #1396)
    - Updated azure pipelines agent configurations (GitHub #1382)
    - Removed obsolete iOSToolchain.cmake (GitHub #1399)

Bug Fixes
~~~~~~~~~
    - Fixed potential double free in Bfr::Surface::operator=() (GitHub #1384)
    - Fixed diagnostic assert in Far::PatchTableFactory (GitHub #1400)
    - Fixed use of deprecated CUDA API (GitHub #1378)
    - Fixed incorrect openmp link dependency when using TBB (GitHub #1375)
