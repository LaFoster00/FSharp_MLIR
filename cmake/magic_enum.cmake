include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

message(STATUS "Downloading and configuring magic_enum")
FetchContent_Declare(
        magic_enum
        GIT_REPOSITORY https://github.com/Neargye/magic_enum
        GIT_TAG v0.9.7
        GIT_PROGRESS TRUE
)
set(MAGIC_ENUM_OPT_INSTALL TRUE)
FetchContent_MakeAvailable(magic_enum)

