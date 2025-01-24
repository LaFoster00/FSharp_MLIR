
let text = "%d\n"

let add a b = a + b

printf text (add 1 2)

let sub a b :float = a - b

let sub_res = sub 1.0 2.5

let add_3 a b c = a + b + c

let add_3_res = add_3 1.0 2.0 3.0

let complex_add a b c =
    let d = a + b
    let e = d + c
    e

let complex_add_res = complex_add 1 2 4

let complex_add_float a b c =
    let d = a + b
    let e = d + c
    e + 2.0

let complex_add_float_res = complex_add_float 1.0 2.0 3.0

let unknown_add a b = a + b
