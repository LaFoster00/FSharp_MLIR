project(Utils)

# Find all .cpp and header files in the current directory
file(GLOB HEADER "${PROJECT_SOURCE_DIR}/utils/*.h")
file(GLOB SOURCE "${PROJECT_SOURCE_DIR}/utils/*.cpp")

add_library(Utils STATIC
        ${SOURCE}
)

target_include_directories(Utils PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Utils>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

set(UTILS_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/Utils")

# Install the static library
install(TARGETS Utils
        EXPORT FSharpMlirTargets       # Add the library to an export set
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}       # For static libraries (e.g., libUtils.a)
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}       # For shared libraries (if converted in the future)
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}       # For executables (if applicable in the future)
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # For include files
)

# Install the header files
install(FILES ${HEADER}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/utils
)

# Export the target for use by other projects
install(EXPORT FSharpMlirTargets
        FILE UtilsTargets.cmake
        NAMESPACE Utils::             # Add a namespace for find_package
        DESTINATION ${UTILS_CMAKE_INSTALL_DIR} # Install the export set
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/UtilsConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/UtilsConfigVersion.cmake"
        DESTINATION ${UTILS_CMAKE_INSTALL_DIR}
)