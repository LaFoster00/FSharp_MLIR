
let d = 10
let e = 6

if (if true then false else true) then
    printf "%d is smaller than %d\n" d e
else
    if true then
        printf "%d is larger than or equal to %d\n" d e
    else
        printf "This should not be printed\n"
