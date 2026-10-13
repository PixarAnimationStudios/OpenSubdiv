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
    - Updated C++ Standard to C++17 (#1388)
    - Updated CMake minimum required and policy max versions to 3.17 and 3.21 respectively (#1393)
    - Updated to use FindCUDAToolkit (#1395)
    - Updated default rpath link behavior (#1376)
    - Updated public constants with inline use of constexpr (#1392)
    - Added alternative constructor for Bfr::Parameterization (#1391)
    - Clarified use of Bfr::Parameterization::Type in documentation (#1397)
    - Improved Bfr::Tessellation quad-preserving patterns (#1386)
    - Added const qualifier to Bfr::Tessellation transform methods (#1385)
    - Addressed compiler warnings (#1387 #1389)
    - Improved processing of compile and link flags (#1390 #1394)
    - Updated recommended package dependencies (#1398)
    - Addressed minor documentation issues (#1396)
    - Updated azure pipelines agent configurations (#1382)
    - Removed obsolete iOSToolchain.cmake (#1399)

Bug Fixes
~~~~~~~~~
    - Fixed potential double free in Bfr::Surface::operator=() (#1384)
    - Fixed diagnostic assert in Far::PatchTableFactory (#1400)
    - Fixed use of deprecated CUDA API (#1378)
    - Fixed incorrect openmp link dependency when using TBB (#1375)
