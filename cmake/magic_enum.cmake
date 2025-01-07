include(FetchContent)

FetchContent_Declare(
        magic_enum
        GIT_REPOSITORY https://github.com/Neargye/magic_enum
        GIT_TAG v0.9.7
)
set(MAGIC_ENUM_OPT_INSTALL TRUE)
FetchContent_MakeAvailable(magic_enum)

