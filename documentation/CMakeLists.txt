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
        cmake_build.rst
        code_examples.rst
        dxviewer.rst
        far_overview.rst
        getting_started.rst
        glviewer.rst
        glbatchviewer.rst
        hbr_overview.rst
        intro.rst 
        limiteval.rst
        osd_overview.rst
        painttest.rst
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
