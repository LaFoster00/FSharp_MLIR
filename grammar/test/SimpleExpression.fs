let a = 10
let b = 2 * -10
let b = 3 * 10 * a
let b = 4 * (a 10)
let b = 5 * (a 10 b)
let b = 6 * ((a 10) + 10)

printf b
printf a a (foo 10) 10 20

bla (10, 20, dergrößte)
bla (10, 20, dergrößte (foo 10))
// NOTE: If dergrößte is a function, it should be called in parantheses otherwise it will be treated an id to a value
bla 10 20 dergrößte (foo 10)

let foo x = x * x
let foo (x: int) : int = x * x
let foo x =
    x * x
