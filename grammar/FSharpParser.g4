parser grammar FSharpParser;

options {
	tokenVocab = FSharpLexer;
}

main: expr+ EOF;

expr: letexpr
    | simpleexpr NEWLINE;

letexpr: LET ID EQUAL simpleexpr NEWLINE;

simpleexpr: callexpr
    | arithexpr
    | literalexpr;

inlineexpr: OPENPAR simpleexpr CLOSEPAR;

literalexpr: INT | STRING;

callexpr: ID callparams;
arithexpr: addexpr | multexpr;
addexpr: INT PLUS (INT | ID);
multexpr: INT STAR (INT | ID);

callparams: OPENPAR (ID | inlineexpr)? CLOSEPAR
    | OPENPAR (ID | simpleexpr) (COMMA (ID | simpleexpr))* CLOSEPAR
    | (ID | inlineexpr | literalexpr) (ID | inlineexpr | literalexpr)*;