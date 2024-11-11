if(NOT UNIX)
    message(WARNING "Unsupported operating system")
endif()

project(
        grammar
        VERSION 0.1
        LANGUAGES CXX C
)

add_subdirectory(demo)