macro(build_antlr)
    message(STATUS "Downloading antlr4 from source.")

    # Define the URL and the destination path
    set(ANTLR_VERSION "4.13.2")
    set(ANTLR_JAR_URL "https://www.antlr.org/download/antlr-${ANTLR_VERSION}-complete.jar")
    set(ANTLR_JAR_LOCATION "${CMAKE_BINARY_DIR}/antlr-${ANTLR_VERSION}-complete.jar" CACHE STRING "")

    # Download the ANTLR jar file if it doesn't exist
    if(NOT EXISTS ${ANTLR_JAR_LOCATION})
        message(STATUS "Downloading ANTLR ${ANTLR_VERSION} jar...")
        file(DOWNLOAD ${ANTLR_JAR_URL} ${ANTLR_JAR_LOCATION}
                SHOW_PROGRESS
                STATUS download_status
        )
        list(GET download_status 0 status_code)
        list(GET download_status 1 error_msg)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "Failed to download ANTLR jar: ${error_msg}")
        endif()
    else()
        message(STATUS "ANTLR jar already downloaded: ${ANTLR_JAR_LOCATION}")
    endif()

    set(ANTLR_EXECUTABLE "${ANTLR_JAR_LOCATION}" CACHE STRING "" FORCE)

    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

    # Specify the version of the antlr4 library needed for this project.
    # By default the latest version of antlr4 will be used.  You can specify a
    # specific, stable version by setting a repository tag value or a link
    # to a zip file containing the libary source.
    set(ANTLR4_TAG 4.13.2)
    # set(ANTLR4_ZIP_REPOSITORY https://github.com/antlr/antlr4/archive/refs/tags/4.13.2.zip)

    # required if linking to static library
    add_definitions(-DANTLR4CPP_STATIC)

    # add external build for antlrcpp
    include(ExternalAntlr4Cpp)

    find_package(ANTLR REQUIRED)
endmacro()