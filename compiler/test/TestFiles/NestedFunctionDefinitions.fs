
let nested_add a b c =
    let internal_add x y = x + y
    internal_add a b + c

printf "Nested add result (6) = %d\n " (nested_add 1 2 3)
