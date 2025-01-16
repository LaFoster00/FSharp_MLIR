include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

message(STATUS "Downloading and configuring Boost")

set(BOOST_INCLUDE_LIBRARIES algorithm)
set(BOOST_ENABLE_CMAKE ON)

# Download and configure Boost
FetchContent_Declare(
        boost
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-1.87.0
        GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(boost)