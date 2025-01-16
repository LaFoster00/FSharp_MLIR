include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

message(STATUS "Downloading and configuring cpptrace")
FetchContent_Declare(
        cpptrace
        GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
        GIT_TAG        v0.7.4 # <HASH or TAG>
        GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(cpptrace)
