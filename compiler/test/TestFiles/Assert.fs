// Integer literal tests
printf "============= Integer literal tests =============\n"
assert (1 = 1) "1 = 1"
assert !(1 = 2) "!(1 = 2)"
assert (1 = 1) "1 = 1"
assert !(1 != 1) "!(1 != 1)"
assert (1 < 2) "1 < 2"
assert !(1 > 2) "!(1 > 2)"
assert (1 <= 1) "1 <= 1"
assert (1 >= 1) "1 >= 1"
assert (1 + 2 = 3) "1 + 2 = 3"
assert (1 - 2 = -1) "1 - 2 = -1"
assert (1 * 2 = 2) "1 * 2 = 2"
assert (1 / 2 = 0) "1 / 2 = 0"
assert (1 % 2 = 1) "1 % 2 = 1"
assert (1 + 2 * 3 = 7) "1 + 2 * 3 = 7"
assert ((1 + 2) * 3 = 9) "((1 + 2) * 3 = 9)"
assert (1 + 2 * 3 - 4 = 3) "1 + 2 * 3 - 4 = 3"
assert (1 + 2 * 3 / 4 = 2) "1 + 2 * 3 / 4 = 2"
assert (1 + 2 * 3 % 4 = 3) "1 + 2 * 3 % 4 = 3"
assert (1 + 2 * 3 % 4 = 3) "1 + 2 * 3 % 4 = 3"

// Float literal tests
printf "============= Float literal tests =============\n"
assert (1.0 = 1.0) "1.0 = 1.0"
assert !(1.0 = 2.0) "!(1.0 = 2.0)"
assert (1.0 = 1.0) "1.0 = 1.0"
assert !(1.0 != 1.0) "!(1.0 != 1.0)"
assert (1.0 < 2.0) "1.0 < 2.0"
assert !(1.0 > 2.0) "!(1.0 > 2.0)"
assert (1.0 <= 1.0) "1.0 <= 1.0"
assert (1.0 >= 1.0) "1.0 >= 1.0"
assert (1.0 + 2.0 = 3.0) "1.0 + 2.0 = 3.0"
assert (1.0 - 2.0 = -1.0) "1.0 - 2.0 = -1.0"
assert (1.0 * 2.0 = 2.0) "1.0 * 2.0 = 2.0"
assert (1.0 / 2.0 = 0.5) "1.0 / 2.0 = 0.5"
assert (1.0 + 2.0 * 3.0 = 7.0) "1.0 + 2.0 * 3.0 = 7.0"
assert ((1.0 + 2.0) * 3.0 = 9.0) "((1.0 + 2.0) * 3.0 = 9.0)"
assert (1.0 + 2.0 * 3.0 - 4.0 = 3.0) "1.0 + 2.0 * 3.0 - 4.0 = 3.0"
assert (1.0 + 2.0 * 3.0 / 4.0 = 2.5) "1.0 + 2.0 * 3.0 / 4.0 = 2.5"
assert (1.0 + 2.0 * 3.0 % 4.0 = 3.0) "1.0 + 2.0 * 3.0 % 4.0 = 3.0"
assert (1.0 + 2.0 * 3.0 % 4.0 = 3.0) "1.0 + 2.0 * 3.0 % 4.0 = 3.0"

// Boolean literal tests
printf "============= Boolean literal tests =============\n"
assert (true && true) "true && true"
assert !(true && false) "!(true && false)"
assert !(false && false) "!(false && false)"
assert (true || false) "true || false"
assert !(false || false) "!(false || false)"

// Function tests
printf "============= Function tests =============\n"
let add a b = a + b
assert ((add 1 2) = 3) "add 1 2 = 3"

let add_3 a b c = a + b + c
assert ((add_3 1.0 2.0 3.0) = 6.0) "add_3 1.0 2.0 3.0 = 6.0"

let complex_add a b c =
    let d = a + b
    let e = d + c
    e
assert ((complex_add 1 2 4) = 7) "complex_add 1 2 4 = 7"

// Nested function test
printf "============= Nested function test =============\n"

let nested_add a b c =
    let internal_add x y = x + y
    internal_add a b + c

assert ((nested_add 1 2 3) = 6) "nested_add 1 2 3 = 6"

// Recursive function test
printf "============= Recursive function test =============\n"
let rec factorial num =
    if num > 1 then
        num * (factorial (num - 1))
    else
        num

assert ((factorial 10) = 3628800) "factorial 10 = 3628800"

// Branching
printf "============= Branching test =============\n"
let and_func x y =
    if x && y then
        true
    else
        false

assert ((and_func true true) = true) "and_func true true = true"
assert ((and_func false true) = false) "and_func false true = false"
assert ((and_func false false) = false) "and_func false false = false"

let and_func_internal_assert x y =
    if x && y then
        assert (x && y) "and_func_internal_assert true true"
    else
        assert (!(x && y)) "and_func_internal_assert false true | false false | true false"

and_func_internal_assert true true
and_func_internal_assert false true
and_func_internal_assert false false
and_func_internal_assert true false
