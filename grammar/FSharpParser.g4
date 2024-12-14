parser grammar FSharpParser;

options {
    superClass = FSharpParserBase;
    tokenVocab = FSharpLexer;
}

@header { #include "FSharpParserBase.h" }

main: (NEWLINE | stmt)* EOF;

stmt
    : simple_stmts
    | compound_stmt
    ;

simple_stmts
    : simple_stmt (SEMI_COLON simple_stmt)* SEMI_COLON? NEWLINE
    ;

simple_stmt
    : expr_stmt
    ;

expr_stmt
    : testlist_expr
    ;

testlist_expr
    : expr (COMMA expr)* COMMA?
    ;

compound_stmt
    : let_stmt
    ;

let_stmt
    : LET name (funcdef | vardef) NEWLINE
    ;

vardef
    : EQUAL test
    ;

funcdef
    : paramlist EQUAL (
        | test test*
    )
    ;

paramlist
    : test (SPACES test)*
    ;

test
    : or_test
    ;

or_test
    : and_test (OR_OP and_test)*
    ;

and_test
    : not_test (AND_OP not_test)*
    ;

not_test
    : EXCLAMATION not_test
    | comparison
    ;

comparison
    : expr (comp_op expr)*
    ;

comp_op
    : LESS_THAN
    | GREATER_THAN
    | EQUAL
    | GT_EQ
    | LT_EQ
    | NOT_EQ
    ;

expr
    : atom
    | (PLUS | MINUS | TILDA)+ expr
    | expr (PLUS | MINUS | STAR | DIV | MOD) expr
    ;

atom_expr
    : atom
    ;

block
     : simple_stmts
     | NEWLINE INDENT stmt+ DEDENT
     ;

atom
    : name
    | NUMBER
    | STRING
    | TRUE
    | FALSE
    ;

name
    : NAME
    | UNDERSCORE
    | MATCH
    ;