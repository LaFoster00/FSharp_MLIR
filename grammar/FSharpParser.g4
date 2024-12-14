parser grammar FSharpParser;

options {
    superClass = FSharpParserBase;
    tokenVocab = FSharpLexer;
}

@header { #include "FSharpParserBase.h" }

main
    : (NEWLINE | stmt)* EOF
    ;

stmt
    : simple_stmts
    //| compound_stmt
    ;

simple_stmts
    : simple_stmt (SEMI_COLON simple_stmt)* SEMI_COLON? NEWLINE
    ;

simple_stmt
    : (
        expr_stmt
        //| flow_stmt
    )
    ;

expr_stmt
    : annassign
    ;


annassign
    : LET name EQUAL test
    ;

flow_stmt
    : break_stmt
    | continue_stmt
    | return_stmt
    ;

break_stmt
    : BREAK
    ;

continue_stmt
    : CONTINUE
    ;

return_stmt
    : RETURN testlist?
    ;

testlist
    : test (',' test)* ','?
    ;

test
    : or_test
    ;

compound_stmt
    : //if_stmt
    //| while_stmt
    //| for_stmt
    //| funcdef
    //| match_stmt
    ;

block
    : simple_stmts
    | NEWLINE INDENT stmt+ DEDENT
    ;

literal_expr
    : signed_number { this->CannotBePlusMinus() }?
    | strings
    | TRUE
    | FALSE
    ;

signed_number
    : NUMBER
    | MINUS NUMBER
    ;

wildcard_pattern
    : UNDERSCORE
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
    : LESS
    | GREATER
    | EQUAL
    | GT_EQ
    | LS_EQ
    | NOT_EQ
    ;

expr
    : atom_expr
    | (PLUSE | MINUS | TILDA)+ expr
    | expr (STAR | DIV | MOD) expr
    | expr (PLUS | MINUS) expr
    ;

atom_expr
    : atom //trailer*
    ;

atom
    : //'(' (yield_expr | testlist_comp)? ')'
    //| '[' testlist_comp? ']'
    //| '{' dictorsetmaker? '}'
    | name
    | NUMBER
    | strings
    | TRUE
    | FALSE
    ;

name
    : NAME
    | UNDERSCORE
    | MATCH
    ;

strings
    : STRING+
    ;
