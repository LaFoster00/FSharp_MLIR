let test_text x y =
    if x && y then
        "x and y are true"
    else
        "x and y are not true"

let a = test_text true true
//let b = test_text false true
//let c = test_text false false

printf "a: %s" a
//printf "b: %A" b
//printf "c: %A" c
