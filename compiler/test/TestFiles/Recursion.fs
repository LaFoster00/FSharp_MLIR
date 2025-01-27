let rec factorial num =
    if num > 1.0 then
        num * (factorial (num - 1.0))
    else
        num

printf "%f" (factorial 10.0)
