using Plots, LaTeXStrings
using LinearAlgebra, ForwardDiff
import ForwardDiff: derivative

f(x) = sin(x)

xrange = range(1.5, 4.5, length=100)

p = plot(xrange, f.(xrange), label=L"$y=\sin(x)$")
x0 = 2
f1(x) = derivative(f, x0) * (x - x0) + f(x0)
x1 = -f(x0)/derivative(f, x0) + x0 

xrange1 = range(x0, x1, length=100)
plot!(xrange1, f1.(xrange1), label="", color="orange")
plot!([x0], seriestype=:vline, label="", color="orange")
plot!([0], seriestype=:hline, label="", color="black")
plot!([x1], seriestype=:vline, label="", color="orange")

x2 = -f(x1)/derivative(f, x1) + x1
f2(x) = derivative(f, x1) * (x - x1) + f(x1)
xrange2 = range(x1, x2, length=100)

plot!(xrange2, f2.(xrange2), label="", color="orange")
vline!([x2], label="", color="orange")

f3(x) = derivative(f, x2) * (x - x2) + f(x2)
x3 = -f(x2)/derivative(f, x2) + x2
xrange3 = range(x2, x3, length=100)
plot!(xrange3, f3.(xrange3), label="", color="orange")

scatter!((x0, 0), label=L"x_0")
scatter!((x1, 0), label=L"x_1", color="yellow")
scatter!((x2, 0), label=L"x_2", color="red")
scatter!((x3, 0), label=L"x_3")

