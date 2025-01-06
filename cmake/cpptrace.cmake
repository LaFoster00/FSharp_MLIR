include(FetchContent)
FetchContent_Declare(
        cpptrace
        GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
        GIT_TAG        v0.7.4 # <HASH or TAG>
)
FetchContent_MakeAvailable(cpptrace)
