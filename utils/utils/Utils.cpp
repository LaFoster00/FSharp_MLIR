//
// Created by lasse on 12/18/24.
//

#include "Utils.h"

#include <sstream>

#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>

namespace utils
{
    std::string demangle(const char* name) {

        int status = -4; // some arbitrary value to eliminate the compiler warning

        // enable c++11 by passing the flag -std=c++11 to g++
        std::unique_ptr<char, void(*)(void*)> res {
            abi::__cxa_demangle(name, NULL, NULL, &status),
            std::free
        };

        return (status==0) ? res.get() : name ;
    }

#else

    // does nothing if not g++
    std::string demangle(const char* name) {
        return name;
    }

#endif
    // Function to add indentation to a multiline string
    std::string indent_string(const std::string& input, const int32_t indent_count) {
        std::istringstream stream(input);
        std::ostringstream result;
        std::string line;
        std::ostringstream indent_stream;
        for (int32_t i = 0; i < indent_count; i++)
        {
            indent_stream << '\t';
        }
        while (std::getline(stream, line)) {
            result << indent_stream.str() << line << '\n'; // Add indentation to each line
        }
        return result.str();
    }

}