let test x y =
    if x && y then
        printf "x and y are true\n"
    else
        printf "x and y are not true\n"

test true true
test false true
test false false
