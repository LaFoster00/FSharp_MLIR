include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

message(STATUS "Downloading and configuring fmt")
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG 11.0.2
        GIT_PROGRESS TRUE
)
set(FMT_INSTALL ON)
FetchContent_MakeAvailable(fmt)