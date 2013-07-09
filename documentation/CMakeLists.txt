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


# Doxygen documentation
if (DOXYGEN_FOUND)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OpenSubdiv.doxy ${CMAKE_CURRENT_BINARY_DIR}/OpenSubdiv.doxy @ONLY)

    add_custom_target(doc_doxy
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/OpenSubdiv.doxy  
        WORKING_DIRECTORY 
            ${CMAKE_BINARY_DIR}/public_headers/
        DEPENDS
            public_headers
        COMMENT 
            "Generating API documentation with Doxygen" VERBATIM
    )

    list(APPEND DOC_TARGETS doc_doxy)

    install( 
        DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}/doxy_html
        DESTINATION 
            ${CMAKE_INSTALL_PREFIX}/documentation
    )

else()

    message(WARNING
        "Doxyen was not found : support for Doxygen automated API documentation is disabled.")

endif()


# ReST - HTML documentation
if (DOCUTILS_FOUND AND PYTHONINTERP_FOUND)

    set(HTML_FILES
         search.html
    )

    set(RST_FILES 
        api_overview.rst
        intro.rst 
        cmake_build.rst
        code_examples.rst
        dxviewer.rst
        getting_started.rst
        glviewer.rst
        glbatchviewer.rst
        limiteval.rst
        ptexviewer.rst
        release_notes.rst 
        subdivision_surfaces.rst 
        using_osd.rst
        using_osd_compile.rst
        using_osd_hbr.rst
        using_osd_textures.rst
    )

    # process rst markup files
    foreach(src ${RST_FILES})
        get_filename_component(BASENAME ${src} NAME_WE)

        set(infile ${CMAKE_CURRENT_SOURCE_DIR}/${src})
        set(outfile ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}.html)

        add_custom_command(
            OUTPUT 
                ${outfile} 
            COMMAND 
                ${RST2HTML_EXECUTABLE}
            ARGS 
                --date
                --time
                --no-xml-declaration 
                --initial-header-level=3
                --strip-comments
                --template=${CMAKE_CURRENT_SOURCE_DIR}/rst_template.txt
                --stylesheet=css/rst.css
                --link-stylesheet
                ${infile} ${outfile} 
            DEPENDS
                ${infile} rst_template.txt nav_template.txt
        )

        add_custom_target(${src} DEPENDS ${outfile})

        list(APPEND RST_TARGETS ${src})

        install( 
            FILES
                ${outfile}
            DESTINATION 
                ${CMAKE_INSTALL_PREFIX}/documentation
            PERMISSIONS 
                OWNER_READ GROUP_READ WORLD_READ 
        )

    endforeach()

    # copy html files to the build area
    foreach(src ${HTML_FILES})
        set(infile ${CMAKE_CURRENT_SOURCE_DIR}/${src})
        set(outfile ${CMAKE_CURRENT_BINARY_DIR}/${src})

        add_custom_command(
            OUTPUT
                ${outfile}
            COMMAND
                ${CMAKE_COMMAND}
            ARGS
                -E copy ${infile} ${outfile}
        )

        add_custom_target( ${src} DEPENDS ${outfile})

        list(APPEND HTML_TARGETS ${src})

        install( 
            FILES
                ${outfile}
            DESTINATION 
                ${CMAKE_INSTALL_PREFIX}/documentation
        )
    endforeach()

    # copy the site resources to the build area so that the 
    # documentation can be read without an install
    add_custom_target(doc_html_images
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/images ${CMAKE_CURRENT_BINARY_DIR}/images
    )

    add_custom_target(doc_html_css
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/css ${CMAKE_CURRENT_BINARY_DIR}/css
    )

    add_custom_target(doc_tipuesearch
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tipuesearch ${CMAKE_CURRENT_BINARY_DIR}/tipuesearch
    )

    # build search index and insert navigation tab
    add_custom_target(search_index
        COMMAND
            ${PYTHON_EXECUTABLE} 
            ${CMAKE_CURRENT_SOURCE_DIR}/processHtml.py 
            ${CMAKE_CURRENT_BINARY_DIR} 
            ${CMAKE_CURRENT_SOURCE_DIR}/nav_template.txt
        DEPENDS 
            ${HTML_TARGETS} 
            ${RST_TARGETS} 
            doc_tipuesearch
    )

    add_custom_target(doc_html 
        DEPENDS 
            search_index 
            ${HTML_TARGETS} 
            ${RST_TARGETS} 
            doc_html_images 
            doc_html_css 
            doc_tipuesearch
    )

    install( 
        DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}/tipuesearch
        DESTINATION 
            ${CMAKE_INSTALL_PREFIX}/documentation
    )


    install( 
        DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}/images
        DESTINATION 
            ${CMAKE_INSTALL_PREFIX}/documentation
    )

    install( 
        DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}/css
        DESTINATION 
            ${CMAKE_INSTALL_PREFIX}/documentation
    )

    list(APPEND DOC_TARGETS doc_html)
endif()

add_custom_target(doc ALL DEPENDS ${DOC_TARGETS})
