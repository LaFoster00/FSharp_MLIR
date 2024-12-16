let i =
    let mutable z = 5 * (a 10 b)
    z <- z * (c 20); z * 2

let a = 10
let c = 2 * -10
let d = 3 * 10 * a
let e = 4 * (a 10)
let f = 5 * (a 10 b c 10)
let g = 6 * ((a 10) + 10)
let h = 5 * (a 10 b) * (c 20)


printf b
printf a a (foo 10) 10 20

module test =
    bla (10, 20, dergrößte)
    bla (10, 20, dergrößte (foo 10))
    //NOTE: If dergrößte is a function, it should be called in parantheses otherwise it will be treated an id to a value
    bla 10 20 dergrößte (foo 10)

    let foo x = x * x
    let foo (x: int) : int = x * x
    let foo x =
        x * x
        x * x

module test2 =
    let foo x = x * x
    let foo (x: int) : int = x * x
    let foo x =
      x * x