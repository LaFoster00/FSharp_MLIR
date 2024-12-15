parser grammar FSharpParser;

options {
    superClass = FSharpParserBase;
    tokenVocab = FSharpLexer;
}

@header { #include "FSharpParserBase.h" }

// Main entry point of the parser
main: (NEWLINE | stmt)* EOF;

// Statement rule, can be either simple or compound
stmt
    : simple_stmts
    | compound_stmt
    ;

// Simple statements followed by a newline
simple_stmts
    : inline_stmts NEWLINE
    ;

// Inline statements separated by semicolons
inline_stmts
    : simple_stmt (SEMI_COLON simple_stmt)* SEMI_COLON?
    ;

// Simple statement, currently only expression statements
simple_stmt
    : expr_stmt
    ;

// Expression statement
expr_stmt
    : testlist_expr
    ;

// List of expressions
testlist_expr
    : test test*
    ;

// Compound statement, currently only let statements
compound_stmt
    : let_stmt
    ;

// Let statement with optional parameters and a block
let_stmt
    : LET name (assignment_stmt | funcdef_stmt)
    ;

assignment_stmt
    : EQUAL block NEWLINE
    ;

funcdef_stmt
    : paramlist EQUAL block NEWLINE
    ;

// List of parameters separated by spaces
paramlist
    : OPEN_PAREN typedarg (COMMA typedarg)* CLOSE_PAREN type?
    | untypedarg (COMMA untypedarg)* type?
    ;

typedarg
    : untypedarg type
    ;

untypedarg
    : name
    ;

type
    : COLON name
    ;


// Test expression, can be one or more expressions
test
    : expr+
    ;

// Logical OR test
or_test
    : and_test (OR_OP and_test)*
    ;

// Logical AND test
and_test
    : not_test (AND_OP not_test)*
    ;

// Logical NOT test
not_test
    : EXCLAMATION not_test
    | comparison
    ;

// Comparison expression
comparison
    : expr (comp_op expr)*
    ;

// Comparison operators
comp_op
    : LESS_THAN
    | GREATER_THAN
    | EQUAL
    | GT_EQ
    | LT_EQ
    | NOT_EQ
    ;

// Expression, can be an atom, unary operation, or binary operation
expr
    : atom_expr
    | (PLUS | MINUS | TILDA)+ expr
    | expr (PLUS | MINUS | STAR | DIV | MOD) expr
    ;

// Atom expression with optional trailers
atom_expr
    : atom trailer*
    ;

// Trailer for atom expressions, currently only dot notation
trailer
    : DOT name
    ;

// Block of statements, can be inline or indented
block
    : inline_stmts
    | NEWLINE INDENT stmt+ DEDENT
    ;

// Atom, can be a parenthesized expression, name, number, string, or boolean
atom
    : OPEN_PAREN exprlist_expr CLOSE_PAREN
    | name
    | NUMBER
    | STRING
    | TRUE
    | FALSE
    ;

// List of atoms
atomlist_expr
    : atom+
    ;

// List of expressions separated by commas
exprlist_expr
    : expr (COMMA? expr)*
    ;

// Name, can be a regular name, underscore, or match keyword
name
    : NAME
    | UNDERSCORE
    | MATCH
    ;