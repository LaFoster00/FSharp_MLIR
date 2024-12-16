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
    | let_stmt #let_definition
    | expr_stmt #expr_definition
    ;

let_stmt
    : LET binding EQUALS body
    ;

binding
    : MUTABLE? ident opt_type #variable_binding
    | REC? ident expr_stmt+ opt_type #standalone_binding
    ;

opt_type
    : (COLON type)?
    ;

body
    : NEWLINE INDENT sequential_expr+ DEDENT #multiline_body
    | sequential_expr #single_line_body
    ;

sequential_expr
    // F# syntax: expr; expr; ...; expr
    : expr_stmt (SEMI_COLON expr_stmt)* NEWLINE
    ;

expr_stmt
    :
    /// F# syntax: 1, 1.3, () etc.
    constant                             #const_expr
    /// F# syntax: ident
    | ident                              #ident_expr
    /// F# syntax: ident.ident...ident
    | long_ident                         #long_ident_expr
    /// F# syntax: ident.ident...ident <- expr
    | long_ident LEFT_ARROW             #long_ident_assign_expr
    /// F# syntax: expr.ident.ident
    | expr_stmt DOT long_ident           #dot_get_expr
    /// F# syntax: expr.ident...ident <- expr
    | expr_stmt DOT long_ident LEFT_ARROW  #dot_set_expr
    /// F# syntax: expr <- expr
    | expr_stmt LEFT_ARROW     #set_expr
    /// F# syntax: expr.[expr]
    | expr_stmt OPEN_BRACK expr_stmt CLOSE_BRACK  #dot_index_get_expr
    /// F# syntax: expr.[expr, ..., expr] <- expr
    | expr_stmt OPEN_BRACK (expr_stmt (COMMA expr_stmt)*)* CLOSE_BRACK LEFT_ARROW  #dot_index_set_expr
    /// F# syntax: null
    | NULL                                          #null_expr
    | expr_stmt operators expr_stmt                           #arith_expr
    | sign expr_stmt                                    #sign_expr

    /// F# syntax: let pat = expr in expr
    /// F# syntax: let f pat1 .. patN = expr in expr
    /// F# syntax: let rec f pat1 .. patN = expr in expr
    /// F# syntax: use pat = expr in expr
    | LET binding EQUALS expr_stmt #let_expr
    /// F# syntax: (expr)
    ///
    /// Parenthesized expressions. Kept in AST to distinguish A.M((x, y))
    /// from A.M(x, y), among other things.
    | OPEN_PAREN expr_stmt CLOSE_PAREN      #paren_expr
    /// F# syntax: expr: type
    | expr_stmt COLON type                        #typed_expr
    /// F# syntax: e1, ..., eN
    | expr_stmt (COMMA expr_stmt)+          #tuple_expr
    /// F# syntax: {| id1=e1; ...; idN=eN |}
    | OPEN_BRACE ident EQUALS expr_stmt (SEMI_COLON ident EQUALS expr_stmt)* CLOSE_BRACE    #anon_record_expr
    /// F# syntax: [ e1; ...; en ]
    | OPEN_BRACK expr_stmt (SEMI_COLON expr_stmt)* CLOSE_BRACK                              #array_expr
    /// F# syntax: [| e1; ...; en |]
    | OPEN_BRACK PIPE expr_stmt (SEMI_COLON expr_stmt)* PIPE CLOSE_BRACK                    #list_expr
    /// F# syntax: new C(...)
    | NEW type OPEN_PAREN expr_stmt (COMMA expr_stmt)* CLOSE_PAREN                          #new_expr
    /// F# syntax: f x
    | expr_stmt expr_stmt+                                                                  #append_expr
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
    | FLOAT
    | STRING
    | CHARACTER
    | BOOL
    | UNIT
    ;