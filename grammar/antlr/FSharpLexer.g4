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

@header {#include "antlr/FSharpLexerBase.h"}

STRING: STRING_LITERAL;
CHARACTER: CHARACTER_LITERAL;
INTEGER: DECIMAL_INTEGER;

UNIT        : '()';
OPEN        : 'open';
NEW         : 'new';
MODULE      : 'module';
NAMESPACE   : 'namespace';
REC         : 'rec';
LET         : 'let';
BREAK       : 'break';
CONTINUE    : 'continue';
IF          : 'if';
THEN        : 'then';
ELIF        : 'elif';
ELSE        : 'else';
BOOL        : 'true' | 'false';
FOR         : 'for';
IN          : 'in';
TO          : 'to';
DO          : 'do';
WHILE       : 'while';
MATCH       : 'match';
MUTABLE     : 'mutable';
UNDERSCORE  : '_';
NULL        : 'null';
WHEN        : 'when';
WITH        : 'with';
AS          : 'as';

NEWLINE: ({this->atStartOfInput()}? WHITESPACES | ( '\r'? '\n' | '\r' | '\f') WHITESPACES?) {this->onNewLine();};

/// identifier   ::=  id_start id_continue*
IDENT: ID_START ID_CONTINUE*;

/// stringliteral   ::=  (shortstring | longstring)
STRING_LITERAL: SHORT_STRING;
CHARACTER_LITERAL: '\'' (STRING_ESCAPE_SEQ | ~[\\\r\n\f'])? '\'';

/// decimalinteger ::=  nonzerodigit digit* | "0"+
DECIMAL_INTEGER: (MINUS | PLUS)? NON_ZERO_DIGIT DIGIT* | '0'+;

/// floatnumber   ::=  pointfloat | exponentfloat
FLOAT_NUMBER: (MINUS | PLUS)? POINT_FLOAT;

DOT                 : '.';
STAR                : '*';
EQUALS              : '=';
PLUS                : '+';
MINUS               : '-';
DIV                 : '/';
MOD                 : '%';
TILDA               : '~';
EXCLAMATION         : '!';
OPEN_PAREN          : '(' {this->openBrace();};
CLOSE_PAREN         : ')' {this->closeBrace();};
OPEN_BRACE          : '{' {this->openBrace();};
CLOSE_BRACE         : '}' {this->closeBrace();};
COMMA               : ',';
COLON               : ':';
SEMI_COLON          : ';';
OPEN_BRACK          : '[' {this->openBrace();};
CLOSE_BRACK         : ']' {this->closeBrace();};
PIPE                : '|';
AMPERCENT           : '&';
AND                 : '&&';
OR                  : '||';
COMPO_LEFT          : '<<';
COMPO_RIGHT         : '>>';
LESS_THAN           : '<';
GREATER_THAN        : '>';
GT_EQ               : '>=';
LT_EQ               : '<=';
NOT_EQ              : '!=';
PIPE_RIGHT          : '|>';
RIGHT_ARROW         : '->';
LEFT_ARROW          : '<-';

SKIP_: (WHITESPACES | COMMENT) -> skip;

UNKNOWN_CHAR: .;

fragment COMMENT: '//' ~[\r\n\f]*;

/// shortstring     ::=  "'" shortstringitem* "'" | '"' shortstringitem* '"'
/// shortstringitem ::=  shortstringchar | stringescapeseq
/// shortstringchar ::=  <any source character except "\" or newline or the quote>
fragment SHORT_STRING: '"' ( STRING_ESCAPE_SEQ | ~[\\\r\n\f"])* '"';

/// stringescapeseq ::=  "\" <any source character>
fragment STRING_ESCAPE_SEQ: '\\' . | '\\' NEWLINE;

/// nonzerodigit   ::=  "1"..."9"
fragment NON_ZERO_DIGIT: [1-9];

/// digit          ::=  "0"..."9"
fragment DIGIT: [0-9];

/// pointfloat    ::=  [intpart] fraction | intpart "."
fragment POINT_FLOAT: INT_PART? FRACTION | INT_PART '.';

/// intpart       ::=  digit+
fragment INT_PART: DIGIT+;

/// fraction      ::=  "." digit+
fragment FRACTION: '.' DIGIT+;

fragment WHITESPACE: [ \t];
fragment WHITESPACES: WHITESPACE+;

// TODO: ANTLR seems lack of some Unicode property support...
//$ curl https://www.unicode.org/Public/13.0.0/ucd/PropList.txt | grep Other_ID_
//1885..1886    ; Other_ID_Start # Mn   [2] MONGOLIAN LETTER ALI GALI BALUDA..MONGOLIAN LETTER ALI GALI THREE BALUDA
//2118          ; Other_ID_Start # Sm       SCRIPT CAPITAL P
//212E          ; Other_ID_Start # So       ESTIMATED SYMBOL
//309B..309C    ; Other_ID_Start # Sk   [2] KATAKANA-HIRAGANA VOICED SOUND MARK..KATAKANA-HIRAGANA SEMI-VOICED SOUND MARK
//00B7          ; Other_ID_Continue # Po       MIDDLE DOT
//0387          ; Other_ID_Continue # Po       GREEK ANO TELEIA
//1369..1371    ; Other_ID_Continue # No   [9] ETHIOPIC DIGIT ONE..ETHIOPIC DIGIT NINE
//19DA          ; Other_ID_Continue # No       NEW TAI LUE THAM DIGIT ONE

fragment UNICODE_OIDS: '\u1885' ..'\u1886' | '\u2118' | '\u212e' | '\u309b' ..'\u309c';

fragment UNICODE_OIDC: '\u00b7' | '\u0387' | '\u1369' ..'\u1371' | '\u19da';


/// id_start     ::=  <all characters in general categories Lu, Ll, Lt, Lm, Lo, Nl, the underscore, and characters with the Other_ID_Start property>
fragment ID_START:
    '_'
    | [\p{L}]
    | [\p{Nl}]
    //| [\p{Other_ID_Start}]
    | UNICODE_OIDS
;

/// id_continue  ::=  <all characters in id_start, plus characters in the categories Mn, Mc, Nd, Pc and others with the Other_ID_Continue property>
fragment ID_CONTINUE:
    ID_START
    | [\p{Mn}]
    | [\p{Mc}]
    | [\p{Nd}]
    | [\p{Pc}]
    //| [\p{Other_ID_Continue}]
    | UNICODE_OIDC
;