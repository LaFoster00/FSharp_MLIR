lexer grammar FSharpLexer;

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

WS: [ \t]+ -> skip;