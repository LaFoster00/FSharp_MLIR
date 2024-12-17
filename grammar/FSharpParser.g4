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
    : MUTABLE? pat? #variable_binding
    | REC? pat? #standalone_binding
    ;

body
    : NEWLINE INDENT sequential_stmt+ DEDENT #multiline_body
    | NEWLINE PIPE sequential_stmt+ #multiline_match_body
    | inline_sequential_stmt #single_line_body
    ;

inline_sequential_stmt
    : expr_stmt (SEMI_COLON expr_stmt)*
    ;

sequential_stmt
    /// F# syntax: expr; expr; ...; expr
    : NEWLINE+
    | expr_stmt (SEMI_COLON expr_stmt)* NEWLINE
    ;

named_pat
    /// F# syntax: ident
    : ident
    ;

arg_pats
    /// F# syntax: pat1 ... patN
    : pat*
    ;

long_ident_pat
    /// F# syntax: ident.ident...ident ident
    : long_ident arg_pats
    ;

typed_pat
    /// F# syntax: pat : type
    : COLON type
    ;

paren_pat
    /// F# syntax: (pat)
    : OPEN_PAREN pat CLOSE_PAREN
    ;

pat //TODO create more fitting version of this that doesnt clutter the ast so much
    // for general pattern matching but with tuple at the top so that it has higher precedence
    : pattern (COMMA pattern)*
    ;

pattern
    :
    /// A constant in a pattern
    constant_expr
    /// A wildcard '_' in a pattern
    | UNDERSCORE
    /// A name pattern 'ident'
    | named_pat
    /// A typed pattern 'pat : type'
    | typed_pat
    /// A disjunctive pattern 'pat1 | pat2'
    | pattern PIPE pattern
    /// A concunctive pattern 'pat1 :: pat2'
    | pattern COLON COLON pattern
    /// A conjunctive pattern 'pat1 & pat2'
    | pattern AMPERCENT pattern
    /// A conjunctive pattern 'pat1 as pat2'
    | pattern AS pattern
    /// A parenthesized pattern
    | paren_pat
    /// Null pattern
    | NULL
    /// A record pattern { identifier1 = pattern_1; ... ; identifier_n = pattern_n }
    | OPEN_BRACE ident EQUALS pattern (SEMI_COLON ident EQUALS pattern)* CLOSE_BRACE
    /// A long identifier pattern possibly with argument patterns
    | long_ident_pat
    ;


dot_get_expr
    /// F# syntax: expr.ident.ident
    : DOT long_ident
    ;

long_ident_assign_expr
    /// F# syntax: ident.ident...ident <- expr
    : long_ident assign_expr
    ;

dot_assign_expr
    /// F# syntax: expr.ident...ident <- expr
    : DOT long_ident assign_expr
    ;

assign_expr
    /// F# syntax: expr <- expr
    : LEFT_ARROW expr_stmt
    ;

dot_index_get_expr
    /// F# syntax: expr.[expr]
    : OPEN_BRACK expr_stmt CLOSE_BRACK
    ;

dot_index_set_expr
    /// F# syntax: expr.[expr, ..., expr] <- expr
    : OPEN_BRACK (expr_stmt (COMMA expr_stmt)*)* CLOSE_BRACK LEFT_ARROW expr_stmt
    ;

arith_expr
    /// F# syntax: expr + expr
    : operators expr_stmt
    ;

signed_expr
    /// F# syntax: (- | +) expr
    : sign expr_stmt
    ;

typed_expr
    /// F# syntax: expr: type
    : COLON type
    ;

tuple_expr
    /// F# syntax: e1, ..., eN
    : COMMA expr_stmt
    ;

paren_expr
    /// F# syntax: (expr)
    : OPEN_PAREN expr_stmt CLOSE_PAREN
    ;

anon_record_expr
    /// F# syntax: {| id1=e1; ...; idN=eN |}
    : OPEN_BRACE ident EQUALS expr_stmt (SEMI_COLON ident EQUALS expr_stmt)* CLOSE_BRACE
    ;

array_expr
    /// F# syntax: [ e1; ...; en ]
    : OPEN_BRACK expr_stmt? (SEMI_COLON expr_stmt)* CLOSE_BRACK
    ;

list_expr
    /// F# syntax: [| e1; ...; en |]
    : OPEN_BRACK PIPE expr_stmt? (SEMI_COLON expr_stmt)* PIPE CLOSE_BRACK
    ;

new_expr
    /// F# syntax: new C(...)
    : NEW type OPEN_PAREN expr_stmt CLOSE_PAREN
    ;

open_expr
    /// F# syntax: open long_ident
    : OPEN long_ident
    ;

comparison_expr
    /// F# syntax: expr > expr
    /// F# syntax: !expr
    : comp_ops expr_stmt
    ;

if_then_else_expr
    /// F# syntax: if expr then body else body
    // The NEWLINE? is needed since body will leave one extra newline
    // This is because we normaly use it to match the sequential_stmt
    : IF expr_stmt THEN body (NEWLINE? ELSE body)?
    ;

match_clause_stmt
    /// F# syntax: | pat -> expr
    : PIPE pat (WHEN expr_stmt)? RIGHT_ARROW body
    ;

match_expr
    /// F# syntax: match expr with | pat1 -> expr1 | ... | patN -> exprN
    : MATCH expr_stmt WITH (NEWLINE? match_clause_stmt )+
    ;

expr_stmt
    :
    expr_stmt expr_stmt
    /// F# syntax: 1, 1.3, () etc.
    | constant_expr
    /// F# syntax: ident
    | ident
    /// F# syntax: ident.ident...ident
    | long_ident
    /// F# syntax: ident.ident...ident <- expr
    | long_ident_assign_expr
    /// F# syntax: expr.ident.ident
    | dot_get_expr
    /// F# syntax: expr.ident...ident <- expr
    | dot_assign_expr
    /// F# syntax: expr <- expr
    | assign_expr
    /// F# syntax: expr.[expr]
    | dot_index_get_expr
    /// F# syntax: expr.[expr, ..., expr] <- expr
    | dot_index_set_expr
    /// F# syntax: let pat = expr in expr
    /// F# syntax: let f pat1 .. patN = expr in expr
    /// F# syntax: let rec f pat1 .. patN = expr in expr
    /// F# syntax: use pat = expr in expr
    | let_stmt
    /// F# syntax: null
    | NULL
    /// F# syntax: expr + expr
    | arith_expr
    /// F# syntax: (- | +) expr
    | signed_expr
    /// F# syntax: expr > expr
    /// F# syntax: !expr
    | comparison_expr
    /// F# syntax: expr: type
    | typed_expr
    /// F# syntax: e1, ..., eN
    | tuple_expr
    /// F# syntax: (expr)
    | paren_expr
    /// F# syntax: {| id1=e1; ...; idN=eN |}
    | anon_record_expr
    /// F# syntax: [ e1; ...; en ]
    | array_expr
    /// F# syntax: [| e1; ...; en |]
    | list_expr
    /// F# syntax: new C(...)
    | new_expr
    /// F# syntax: open long_ident
    | open_expr
    | if_then_else_expr
    | match_expr
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


append_bracket_generic_type
    /// F# syntax: type<type, ..., type>
    : LESS_THAN type (COMMA type)* GREATER_THAN
    ;

postfix_double_type
    /// F# syntax: (type, type) type
    : OPEN_PAREN type COMMA type CLOSE_PAREN type
    ;

long_ident_append_type
    /// F# syntax: type.A.B.C<type, ..., type>
    : (DOT ident)* LESS_THAN type (COMMA type)* GREATER_THAN
    ;

tuple_type
    /// F# syntax: type * ... * type
    : (STAR type)+
    ;

array_type
    /// F# syntax: type[]
    : OPEN_BRACK CLOSE_BRACK
    ;

fun_type
    /// F# syntax: type -> type
    : RIGHT_ARROW type
    ;

paren_type
    /// F# syntax: (type)
    : OPEN_PAREN type CLOSE_PAREN
    ;

type:
    /// F# syntax: type type
    type type
    /// F# syntax: type
    | ident

    /// F# syntax: A.B.C
    | long_ident

    /// F# syntax: type<type, ..., type>
    | append_bracket_generic_type

    /// F# syntax: (type, type) type
    | postfix_double_type

    /// F# syntax: type.A.B.C<type, ..., type>
    | long_ident_append_type

    /// F# syntax: type * ... * type
    | tuple_type

    /// F# syntax: type[]
    | array_type

    /// F# syntax: type -> type
    | fun_type

    /// F# syntax: (type)
    | paren_type

    ;

ident
    : IDENT
    ;

constant_expr
    : INTEGER
    | FLOAT_NUMBER
    | STRING
    | CHARACTER_LITERAL
    | BOOL
    | UNIT
    ;
