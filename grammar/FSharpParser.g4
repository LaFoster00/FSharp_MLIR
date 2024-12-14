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
    : inline_stmts NEWLINE
    ;

inline_stmts
    : simple_stmt (SEMI_COLON simple_stmt)* SEMI_COLON?
    ;

simple_stmt
    : expr_stmt
    ;

expr_stmt
    : testlist_expr
    ;

testlist_expr
    : test test*
    ;

compound_stmt
    : let_stmt
    ;

let_stmt
    : LET name paramlist? EQUAL block NEWLINE
    ;

paramlist
    : test (SPACES test)*
    ;

test
    : expr+
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
    : atom_expr
    | (PLUS | MINUS | TILDA)+ expr
    | expr (PLUS | MINUS | STAR | DIV | MOD) expr
    ;

atom_expr
    : atom trailer*
    ;

trailer
    : DOT name
    ;

block
     : inline_stmts
     | NEWLINE INDENT stmt+ DEDENT
     ;

atom
    : OPEN_PAREN exprlist_expr CLOSE_PAREN
    | name
    | NUMBER
    | STRING
    | TRUE
    | FALSE
    ;

atomlist_expr
    : atom+
    ;

exprlist_expr
    : expr (COMMA? expr)*
    ;

name
    : NAME
    | UNDERSCORE
    | MATCH
    ;