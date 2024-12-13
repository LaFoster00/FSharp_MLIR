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

literalexpr: INT
    | STRING;

callexpr: ID callparams;

arithexpr: addexpr
    | multexpr;

operandsexpr: | ID
    | literalexpr
    | inlineexpr;

addexpr: operandsexpr PLUS (operandsexpr | ID);

multexpr: operandsexpr STAR (operandsexpr | ID);

callparams:
    OPENPAR (ID | inlineexpr)? CLOSEPAR
    | OPENPAR (ID | simpleexpr | inlineexpr) (COMMA (ID | simpleexpr | inlineexpr))* CLOSEPAR
    | (ID | literalexpr) (ID | inlineexpr | literalexpr)*
    ;