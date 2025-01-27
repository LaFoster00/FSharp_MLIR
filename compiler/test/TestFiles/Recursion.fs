let rec factorial num =
    if num > 1 then
        num * (factorial (num - 1))
    else
        num

printf "%i" (factorial 10)
