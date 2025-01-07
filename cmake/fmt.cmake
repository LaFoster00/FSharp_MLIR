include(FetchContent)

FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG 11.0.2
)
set(FMT_INSTALL ON)
FetchContent_MakeAvailable(fmt)