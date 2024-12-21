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
    std::string indent_string(const std::string& input, const int32_t indent_count, bool addParen, bool lastParenNewLine, bool trailingNewline) {
        std::istringstream stream(input);
        std::ostringstream result;
        std::string line;
        std::ostringstream indent_stream;
        for (int32_t i = 0; i < indent_count; i++)
        {
            indent_stream << '\t';
        }
        auto indent_string = indent_stream.str();

        bool first_line = true;
        while (std::getline(stream, line)) {
            if (!line.empty())
            {
                if (first_line)
                {
                    first_line = false;
                    result << indent_string;
                    if (addParen)
                        result << '(';
                    result << line << '\n';
                }
                else
                {
                    result << indent_string << line << '\n'; // Add indentation to each line
                }
            }
        }

        // Place the closing bracket
        auto string = result.str();
        if (addParen && !lastParenNewLine && string.back() == '\n')
            string.pop_back();

        if (addParen)
            string += (lastParenNewLine ? indent_string : "") + ')';

        if (trailingNewline)
            string += '\n';

        return string;
    }

}