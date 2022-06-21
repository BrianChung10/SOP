using Plots, LaTeXStrings


f(x) = sqrt(x)
f1(x) = 0
f2(x) = sqrt(x) + 0.5
f3(x) = 0.5
xrange = range(0, 10, length=200)
xrange1 = range(-2.5, 2.5, length=200)
xrange2 = range(0, 2.5, length=200)
xrange3 = range(-2.5, 0, length=200)
xrange4 = range(0, 5, length=200)


p = plot(xrange, f2.(xrange), color="black", legend=:topleft, label="")
plot!(xrange, -f.(xrange), color="black", label="")
plot!(xrange, f3.(xrange), color="black", label="")
plot!(xrange3, f1.(xrange3), color="red", label="")
plot!(xrange4, -f.(xrange4), color="red", label="", arrow=true)
vline!([2.5], label="", color="green")
r1 = (2.5, -f(2.5))
r2 = (2.5, 0.5)
r3 = (2.5, f2(2.5))
scatter!(r1, label="", color="blue")
scatter!(r2, label="", color="blue")
scatter!(r3, label="", color="blue")

xrange5 = range(2.5, 5, length=200)
plot!(xrange5, f2.(xrange5), label="", color="red", arrow=true)
plot!(xrange5, f3.(xrange5), label="", color="red", arrow=true)

xrange6 = range(2.5, 1, length=200)
plot!(xrange6, f2.(xrange6), label="", color="red", arrow=true)
plot!(xrange6, f3.(xrange6), label="", color="red", arrow=true)