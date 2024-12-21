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
    | sequential_stmt #expression_stmt
    | OPEN long_ident #open_stmt
    ;

inline_sequential_stmt
    : expression (SEMI_COLON expression)*
    ;

sequential_stmt
    /// F# syntax: expr; expr; ...; expr
    : NEWLINE+
    | expression (SEMI_COLON expression)* NEWLINE
    ;

/// PATERN MATCHING
// Patern matching follows this structure to give a clear precedence of the patterns
// Having them all in one production will cause bad matching behaviour with non trivial asts
pattern
    :
    tuple_pat
    ;

tuple_pat
    /// F# syntax: pat, ..., pat
    : and_pat (COMMA and_pat)*
    ;

and_pat
    /// F# syntax: pat1 & pat2
    : or_pat (AMPERCENT or_pat)*
    ;

or_pat
    /// F# syntax: pat1 | pat2
    : as_pat (PIPE as_pat)*
    ;

as_pat
    /// F# syntax: pat1 as pat2
    : cons_pat (AS cons_pat)?
    ;

cons_pat
    /// F# syntax: pat1 :: pat2
    : typed_pat (COLON COLON typed_pat)?
    ;

typed_pat
    /// F# syntax: pat : type
    : atomic_pat (COLON type)?
    ;

atomic_pat
    :
    paren_pat
    | anon_expr
    | constant
    | named_pat
    | record_pat
    | array_pat
    | long_ident_pat
    | null_pat
    ;

paren_pat
    /// F# syntax: (pat)
    : OPEN_PAREN pattern CLOSE_PAREN
    ;

record_pat
    /// F# syntax: { id1 = pat1; ...; idN = patN }
    : OPEN_BRACE ident EQUALS atomic_pat (SEMI_COLON ident EQUALS atomic_pat)* CLOSE_BRACE
    ;

array_pat
    /// F# syntax: [pat1; ...; patN]
    : OPEN_BRACK atomic_pat? (SEMI_COLON atomic_pat)* CLOSE_BRACK
    ;

named_pat
    /// F# syntax: ident
    : ident
    ;

anon_expr
    /// F# syntax: _
    : UNDERSCORE
    ;

null_pat
    /// F# syntax: null
    : NULL
    ;

long_ident_pat
   /// F# syntax: A.B pat1 ... patN
   : long_ident atomic_pat*
   ;

/// EXPRESSIONS
expression
    :
    assignment_expr
    | non_assigment_expr
    ;

assignment_expr
    : let_expr
    | long_ident_set_expr
    | set_expr
    | dot_set_expr
    | dot_index_set_expr
    ;

let_expr
    : LET binding EQUALS body
    ;

binding
    : (MUTABLE? | REC?) pattern?
    ;

body
    : NEWLINE INDENT sequential_stmt+ DEDENT #multiline_body
    | NEWLINE PIPE sequential_stmt+ #multiline_match_body
    | inline_sequential_stmt #single_line_body
    ;

long_ident_set_expr
    /// F# syntax: ident.ident...ident <- expr
    : long_ident LEFT_ARROW expression
    ;

set_expr
    /// F# syntax: expr <- expr
    : atomic_expr LEFT_ARROW expression
    ;

dot_set_expr
    /// F# syntax: expr.ident...ident <- expr
    : atomic_expr DOT long_ident LEFT_ARROW expression
    ;

dot_index_set_expr
    /// F# syntax: expr.[expr, ..., expr] <- expr
    : atomic_expr OPEN_BRACK (expression (COMMA expression)*)* CLOSE_BRACK LEFT_ARROW expression
    ;

non_assigment_expr
    : app_expr
    ;

app_expr
    : tuple_expr tuple_expr*
    ;

tuple_expr
    /// F# syntax: e1, ..., eN
    : or_expr (COMMA or_expr)*
    ;

or_expr
    /// F# syntax: expr | expr
    : and_expr (PIPE and_expr)*
    ;

and_expr
    /// F# syntax: expr & expr
    : equality_expr (AMPERCENT equality_expr)*
    ;

equality_expr
    /// F# syntax: expr = expr
    : relation_expr ((EQUALS | NOT_EQ) relation_expr)*
    ;

relation_expr
    /// F# syntax: expr > expr
    : additive_expr ((GREATER_THAN | LESS_THAN | GT_EQ | LT_EQ) additive_expr)*
    ;

additive_expr
    /// F# syntax: expr + expr
    : multiplicative_expr ((PLUS | MINUS) multiplicative_expr)*
    ;

multiplicative_expr
    /// F# syntax: expr * expr
    : dot_get_expr ((STAR | DIV | MOD) dot_get_expr)*
    ;

dot_get_expr
    /// F# syntax: expr.ident.ident
    : dot_index_get_expr (DOT long_ident)?
    ;

dot_index_get_expr
    /// F# syntax: expr.[expr]
    :  typed_expr (DOT OPEN_BRACK typed_expr CLOSE_BRACK)?
    ;

typed_expr
    /// F# syntax: expr: type
    : unary_expression (COLON type)?
    ;

unary_expression
    :
    atomic_expr
    | PLUS unary_expression
    | MINUS unary_expression
    | EXCLAMATION unary_expression
    ;

atomic_expr
    :
    paren_expr
    | constant_expr
    | ident_expr
    | long_ident_expr
    | null_expr
    | record_expr
    | array_expr
    | list_expr
    | new_expr
    | if_then_else_expr
    | match_expr
    | pipe_right_expr
    ;

constant_expr
     : constant
     ;

ident_expr
    : ident
    ;

long_ident_expr
    : long_ident
    ;

null_expr
    /// F# syntax: null
    : NULL
    ;

paren_expr
    /// F# syntax: (expr)
    : OPEN_PAREN expression CLOSE_PAREN
    ;

record_expr
    /// F# syntax: {| id1=e1; ...; idN=eN |}
    : OPEN_BRACE record_expr_field (SEMI_COLON record_expr_field)* CLOSE_BRACE
    ;

record_expr_field
    :
    /// F# syntax: id = expr
    ident EQUALS expression
    ;

array_expr
    /// F# syntax: [ e1; ...; en ]
    : OPEN_BRACK expression? (SEMI_COLON expression)* CLOSE_BRACK
    ;

list_expr
    /// F# syntax: [| e1; ...; en |]
    : OPEN_BRACK PIPE expression? (SEMI_COLON expression)* PIPE CLOSE_BRACK
    ;

new_expr
    /// F# syntax: new C(...)
    : NEW type OPEN_PAREN expression CLOSE_PAREN
    ;

if_then_else_expr
    /// F# syntax: if expr then body else body
    // The NEWLINE? is needed since body will leave one extra newline
    // This is because we normaly use it to match the sequential_stmt
    : IF expression THEN body (NEWLINE? ELSE body)?
    ;

match_expr
    /// F# syntax: match expr with | pat1 -> expr1 | ... | patN -> exprN
    : MATCH expression WITH (NEWLINE? match_clause_stmt )+
    ;

pipe_right_expr
    /// F# syntax: |> expr
    : PIPE_RIGHT body
    ;

match_clause_stmt
    /// F# syntax: | pat -> expr
    : PIPE pattern (WHEN expression)? RIGHT_ARROW body
    ;


operators
    : PLUS
    | MINUS
    | STAR
    | DIV
    | MOD
    ;

comp_ops
    : EQUALS
    | GREATER_THAN
    | LESS_THAN
    | GT_EQ
    | LT_EQ
    | NOT_EQ
    | EXCLAMATION
    ;

sign
    : PLUS
    | MINUS
    ;

long_ident
    : ident (DOT ident)*;


type:
    fun_type
    ;

fun_type
    /// F# syntax: type -> type
    : tuple_type (RIGHT_ARROW type)*
    ;

tuple_type
    /// F# syntax: type * ... * type
    : append_type (STAR type)*
    ;

append_type
    /// F# syntax: type<type, ..., type> or type type or (type, ..., type) type
    : long_ident_append_type generic_args? #generic_type
    | long_ident_append_type long_ident_append_type? #postfix_type
    | paren_type long_ident_append_type #paren_postfix_type
    ;

long_ident_append_type
    /// F# syntax: type.A.B.C<type, ..., type>
    :
    array_type ((DOT long_ident)+ generic_args)?
    ;

generic_args
    /// F# syntax: <type, ..., type>
    : LESS_THAN type (COMMA type)* GREATER_THAN
    ;

array_type
    /// F# syntax: type[]
    : atomic_type (OPEN_BRACK CLOSE_BRACK)?
    ;



atomic_type
    : paren_type
    | var_type
    | long_ident
    | anon_type
    | static_constant_type
    | static_constant_null_type
    | generic_args
    ;

paren_type
    /// F# syntax: (type)
    : OPEN_PAREN type CLOSE_PAREN
    ;

var_type
    /// F# syntax: var
    : IDENT
    ;

anon_type
    /// F# syntax: _
    : UNDERSCORE
    ;

static_constant_type
    : constant
    ;

static_constant_null_type
    : constant
    | NULL
    ;


ident
    : IDENT
    ;

constant
    : INTEGER
    | FLOAT_NUMBER
    | STRING
    | CHARACTER
    | BOOL
    | UNIT
    ;
