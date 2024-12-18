//
// Created by lasse on 12/18/24.
//
#pragma once

#include <string>

namespace utils
{
    // Using utils::to_string allows us to create to_string overrides in our classes
    namespace adl_helper {
        template<typename T>
        std::string as_string( T&& t ) {
            using std::to_string;
            return to_string( std::forward<T>(t) );
        }
    }
    template<typename T>
    std::string to_string( T&& t ) {
        return adl_helper::as_string(std::forward<T>(t));
    }
} // namespace utils
