
let a = 1 + 2 - 3 * 4 / 5
printf "result int: ist %d, soll 1\n" a

let b = 1.0 + 2.0 - 3.0 * 4.0 / 5.0
printf "result float: ist %f, soll 0.6\n" b

let c = 1.0 + 2 - 3 * 4 / 5
printf "result mixed: ist %f, soll 0.6\n" c

let d = 10 % 3
printf "result modulo: ist %d, soll 1\n" d

let d2 = 10.0 % 3.0
printf "result modulo float: ist %f, soll 1.0\n" d2
