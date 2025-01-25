let test x y =
    if x && y then
        printf "x and y are true"
    else
        printf "x and y are not true"

test true true
test false true
test false false
