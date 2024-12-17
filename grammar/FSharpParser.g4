parser grammar FSharpParser;

options {
    superClass = FSharpParserBase;
    tokenVocab = FSharpLexer;
}

@header { #include "FSharpParserBase.h" }

main: (NEWLINE | module_or_namespace)+ EOF;

module_or_namespace
    : MODULE long_ident NEWLINE module_decl*    #namedmodule
    | NAMESPACE long_ident NEWLINE module_decl* #namespace
    | module_decl+                              #anonmodule
    ;

/// Represents a definition within a module
module_decl
    : NEWLINE+ #emply_lines
    | MODULE long_ident EQUALS NEWLINE INDENT module_decl* DEDENT #nested_module
    | sequential_stmt #expr_definition
    ;

let_stmt
    : LET binding EQUALS body
    ;

binding
    : MUTABLE? ident opt_type #variable_binding
    | REC? ident expr_stmt opt_type #standalone_binding
    ;

opt_type
    : (COLON type)?
    ;

body
    : NEWLINE INDENT sequential_stmt+ DEDENT #multiline_body
    | inline_sequential_stmt #single_line_body
    ;

dot_get
    /// F# syntax: expr.ident.ident
    : DOT long_ident
    ;

long_ident_assign
    /// F# syntax: ident.ident...ident <- expr
    : long_ident assign
    ;

dot_assign
    /// F# syntax: expr.ident...ident <- expr
    : DOT long_ident assign
    ;

assign
    /// F# syntax: expr <- expr
    : LEFT_ARROW expr_stmt
    ;

dot_index_get
    /// F# syntax: expr.[expr]
    : OPEN_BRACK expr_stmt CLOSE_BRACK
    ;

dot_index_set
    /// F# syntax: expr.[expr, ..., expr] <- expr
    : OPEN_BRACK (expr_stmt (COMMA expr_stmt)*)* CLOSE_BRACK LEFT_ARROW expr_stmt
    ;

arith
    /// F# syntax: expr + expr
    : operators expr_stmt
    ;

signed
    /// F# syntax: (- | +) expr
    : sign expr_stmt
    ;

typed
/// F# syntax: expr: type
    : COLON type
    ;

tuple
    /// F# syntax: e1, ..., eN
    : (COMMA expr_stmt)+ // TODO check if + here is helping
    ;

paren
    /// F# syntax: (expr)
    : OPEN_PAREN expr_stmt CLOSE_PAREN
    ;

anon_record
    /// F# syntax: {| id1=e1; ...; idN=eN |}
    : OPEN_BRACE ident EQUALS expr_stmt (SEMI_COLON ident EQUALS expr_stmt)* CLOSE_BRACE
    ;

array
    /// F# syntax: [ e1; ...; en ]
    : OPEN_BRACK expr_stmt? (SEMI_COLON expr_stmt)* CLOSE_BRACK
    ;

list
    /// F# syntax: [| e1; ...; en |]
    : OPEN_BRACK PIPE expr_stmt? (SEMI_COLON expr_stmt)* PIPE CLOSE_BRACK
    ;

new
    /// F# syntax: new C(...)
    : NEW type OPEN_PAREN expr_stmt CLOSE_PAREN
    ;

open
    /// F# syntax: open long_ident
    : OPEN long_ident
    ;

inline_sequential_stmt
    : expr_stmt (SEMI_COLON expr_stmt)*
    ;

sequential_stmt
    /// F# syntax: expr; expr; ...; expr
    : expr_stmt (SEMI_COLON expr_stmt)* NEWLINE
    ;


expr_stmt
    :

    expr_stmt expr_stmt #append_expr
    /// F# syntax: 1, 1.3, () etc.
    | constant #const_expr
    /// F# syntax: ident
    | ident #ident_expr
    /// F# syntax: ident.ident...ident
    | long_ident #long_ident_expr
    /// F# syntax: ident.ident...ident <- expr
    | long_ident_assign #long_ident_assign_expr
    /// F# syntax: expr.ident.ident
    | dot_get #dot_get_expr
    /// F# syntax: expr.ident...ident <- expr
    | dot_assign  #dot_set_expr
    /// F# syntax: expr <- expr
    | assign    #set_expr
    /// F# syntax: expr.[expr]
    | dot_index_get  #dot_index_get_expr
    /// F# syntax: expr.[expr, ..., expr] <- expr
    | dot_index_set #dot_index_set_expr
    /// F# syntax: let pat = expr in expr
    /// F# syntax: let f pat1 .. patN = expr in expr
    /// F# syntax: let rec f pat1 .. patN = expr in expr
    /// F# syntax: use pat = expr in expr
    | let_stmt #let_expr
    /// F# syntax: null
    | NULL #null_expr
    | arith #arith_expr
    | signed  #sign_expr
    /// F# syntax: expr: type
    | typed #typed_expr
    /// F# syntax: e1, ..., eN
    | tuple #tuple_expr
    /// F# syntax: (expr)
    | paren #paren_expr
    /// F# syntax: {| id1=e1; ...; idN=eN |}
    | anon_record #anon_record_expr
    /// F# syntax: [ e1; ...; en ]
    | array #array_expr
    /// F# syntax: [| e1; ...; en |]
    | list #list_expr
    /// F# syntax: new C(...)
    | new #new_expr
    /// F# syntax: open long_ident
    | open #open_expr
    ;

operators
    : PLUS
    | MINUS
    | STAR
    | DIV
    | MOD
    ;

sign
    : PLUS
    | MINUS
    ;

long_ident
    : ident (DOT ident)*;


type:
    /// F# syntax: A.B.C
    long_ident                                            #long_ident_type

    // | APPEND productions
    /// F# syntax: type<type, ..., type>
    | type LESS_THAN (type (COMMA type)*)* GREATER_THAN       #append_bracket_generic_type
    /// F# syntax: type type
    | type type                                             #append_double_type
    /// F# syntax: (type, type) type
    | OPEN_PAREN type COMMA type CLOSE_PAREN type                 #postfix_double_type

    //| long_ident_append
    /// F# syntax: type.A.B.C<type, ..., type>
    | type (DOT ident)* LESS_THAN type (COMMA type)* GREATER_THAN    #long_ident_append_type

    //| tuple
    /// F# syntax: type * ... * type
    | type (STAR type)+                                    #tuple_type

    //| array
    /// F# syntax: type[]
    | type OPEN_BRACK CLOSE_BRACK                           #array_type

    //| fun
    /// F# syntax: type -> type
    | type ARROW type                                       #fun_type

    //| paren
    /// F# syntax: (type)
    | OPEN_PAREN type CLOSE_PAREN                           #paren_type

    ;

ident
    : IDENT
    ;

constant
    : INTEGER
    | FLOAT_NUMBER
    | STRING
    | CHARACTER_LITERAL
    | BOOL
    | UNIT
    ;
