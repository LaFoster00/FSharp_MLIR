lexer grammar FSharpLexer;

// https://github.com/antlr/grammars-v4/tree/master/python/python3

// All comments that start with "///" are copy-pasted from
// The Python Language Reference

tokens {
    INDENT,
    DEDENT
}

options {
    superClass = FSharpLexerBase;
}

channels { CommentsChannel }

@header {#include "FSharpLexerBase.h"}


LET: 'let';

INT: Digit+;
Digit: [0-9];

ID: LETTER (LETTER | '0'..'9')*;
fragment LETTER : [a-zA-Z\u0080-\u{10FFFF}];

EQUAL: '=';
PLUS: '+';
STAR: '*';
OPENPAR: '(';
CLOSEPAR: ')';
COMMA: ',';

STRING: '"' .*? '"';

NEWLINE: '\r'? '\n';

COMMENT : '//' ~[\r\n]* '\r'? '\n' -> channel(CommentsChannel);
WS: [ \t]+ -> skip;