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
    include(${PROJECT_SOURCE_DIR}/extern/antlr4/runtime/Cpp/cmake/FindANTLR.cmake)


    # Compile antlr4 java runtime
    message(STATUS "Compiling antlr4 CPP runtime from source.")
    set(WITH_DEMO FALSE CACHE BOOL "")
    set(WITH_LIBCXX TRUE CACHE BOOL "")
    add_subdirectory(extern/antlr4/runtime/Cpp)
endmacro()