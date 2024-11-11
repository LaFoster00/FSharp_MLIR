# -*- mode:cmake -*-
if(NOT UNIX)
  message(WARNING "Unsupported operating system")
endif()


message(STATUS ${PROJECT_SOURCE_DIR})

set(antlr4-demo-GENERATED_SRC
   ${PROJECT_SOURCE_DIR}/demo/generated/TLexer.cpp 
   ${PROJECT_SOURCE_DIR}/demo/generated/TParser.cpp
   ${PROJECT_SOURCE_DIR}/demo/generated/TParserBaseListener.cpp
   ${PROJECT_SOURCE_DIR}/demo/generated/TParserBaseVisitor.cpp
   ${PROJECT_SOURCE_DIR}/demo/generated/TParserListener.cpp
   ${PROJECT_SOURCE_DIR}/demo/generated/TParserVisitor.cpp
 )

foreach(src_file ${antlr4-demo-GENERATED_SRC})
      set_source_files_properties(
          ${src_file}
          PROPERTIES
          GENERATED TRUE
          )
endforeach(src_file ${antlr4-demo-GENERATED_SRC})

message(STATUS "Generating Parser")
add_custom_target(GenerateParser DEPENDS ${antlr4-demo-GENERATED_SRC})
add_custom_command(OUTPUT ${antlr4-demo-GENERATED_SRC}
   COMMAND 
   ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/demo/generated/
   COMMAND
   "${Java_JAVA_EXECUTABLE}" -jar ${ANTLR_JAR_LOCATION} -Werror -Dlanguage=Cpp -listener -visitor -o ${PROJECT_SOURCE_DIR}/demo/generated/ -package antlrcpptest ${PROJECT_SOURCE_DIR}/demo/TLexer.g4 ${PROJECT_SOURCE_DIR}/demo/TParser.g4
   WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
   DEPENDS ${PROJECT_SOURCE_DIR}/demo/TLexer.g4 ${PROJECT_SOURCE_DIR}/demo/TParser.g4
   )

include_directories(
  ${PROJECT_SOURCE_DIR}/demo/generated
  )

#file(GLOB antlr4-demo_SRC "${PROJECT_SOURCE_DIR}/demo/generated/*")
set(antlr4-demo_SRC 
  ${PROJECT_SOURCE_DIR}/demo/main.cpp
  ${antlr4-demo-GENERATED_SRC}
  )

set (flags_1 "-Wno-overloaded-virtual")

foreach(src_file ${antlr4-demo_SRC})
      set_source_files_properties(
          ${src_file}
          PROPERTIES
          COMPILE_FLAGS "${COMPILE_FLAGS} ${flags_1}"
          )
endforeach(src_file ${antlr4-demo_SRC})

message(STATUS "Adding antlr4 demo")
add_executable(antlr4-demo
    ${antlr4-demo_SRC}
    ${ANTLR_RUNTIME_}
)
#add_precompiled_header(antlr4-demo ${PROJECT_SOURCE_DIR}/runtime/src/antlrcpp-Prefix.h)

add_dependencies(antlr4-demo GenerateParser)

target_link_libraries(antlr4-demo antlr4_static)

install(TARGETS antlr4-demo 
        DESTINATION "share" 
        COMPONENT dev 
        )
