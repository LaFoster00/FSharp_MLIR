
let test x y = 
    if x and y then
        printf "x and y are true"
    else
        printf "x and y are not true"

let a = test true true
let b = test false true
let c = test false false

printf "a: %A" a
printf "b: %A" b
printf "c: %A" c
