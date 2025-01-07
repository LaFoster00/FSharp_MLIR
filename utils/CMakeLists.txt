project(Utils)

# Find all .cpp and header files in the current directory
file(GLOB HEADER "${PROJECT_SOURCE_DIR}/utils/*.h")
file(GLOB SOURCE "${PROJECT_SOURCE_DIR}/utils/*.cpp")

add_library(Utils STATIC
        ${SOURCE}
)

target_include_directories(Utils PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
)

install(TARGETS Utils
        EXPORT UtilsTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
        EXPORT UtilsTargets
        FILE UtilsTargets.cmake
        NAMESPACE Utils::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
)