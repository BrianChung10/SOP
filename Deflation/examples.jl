using Plots
using ForwardDiff, LinearAlgebra, LaTeXStrings
import ForwardDiff: derivative, jacobian
import LinearAlgebra: norm, inv, I


function M(x, x1, p=2, alpha=1)
    1 / norm(x-x1)^p + alpha
end


function newton(f, x0, max_iter=1000, eps=1e-8)
    x = x0
    i = 0
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge."
        end
        A = jacobian(F, x)
        x = x - (qr(A) \ f(x))  # do least square fit which is more robust
        i += 1
    end
    x
end


function deflated_newton(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton(g, x0)
end


# Example 1: The Cubic-Parabola
f(x, y) = 5x^3 - 4x - y
g(x, y) = 3x^2 + x - y

f1(x) = 5x^3 - 4x
g1(x) = 3x^2 + x

p = plot(f1, -1.5, 1.5, label=L"5x^3-4x-y=0")
plot!(g1, -1.5, 1.5, label=L"3x^2+x-y=0")


x0 = [0.5; 0.5] # Initial guess
h1(x) = [f(x[1], x[2]); g(x[1], x[2])] # Original function (Cubic-Parabola)
x1 = newton(h1, x0) # Find the first root
h2(x) = M(x, x1) * h1(x) # Define the deflated function
x2 = newton(h2, x0) # Find the second root with the same intial guess
h3(x) = M(x, x2) * h2(x) # Define the second deflated function
x3 = newton(h3, x0)
sol = [x1, x2, x3]

# Define the function F
F(x) = [5x[1]^3 - 4x[1] - x[2]; 3x[1]^2 + x[1] - x[2]]

# Define a function to test whether r = (0, 0) is a magnetic zero
function mag_zero()
    success = 0
    fail = 0
    for i = -1: 0.1: 1
        for j = -1: 0.1: 1
            x0 = [i; j]
            x1 = newton(F, x0)
            if norm(x1 - [0; 0]) < 1e-10
                success += 1
            else
                fail += 1
            end
        end
    end
    println(success)
    println(fail)
end

mag_zero()

# The four-cluster system
F(x) = [(x[1] - x[2]^2) * (x[1] - sin(x[2])); (cos(x[2]) - x[1]) * (x[2] - cos(x[1]))]
# f(x, y) = (x - y^2) * (x - sin(y))
# g(x, y) = (cos(y) - x) * (y - cos(x))

f1(x) = x^2
f2(x) = sin(x)
f3(x) = cos(x)

xrange = range(0, 2, length=200)
xrange2 = range(0, pi/2, length=200)
xrange3 = range(0, 1.25, length=100)
p = plot(f1.(xrange3), xrange3, label=L"x = y^2")
plot!(f2.(xrange), xrange, label=L"x = \sin(y)")
plot!(f3.(xrange2), xrange2, label=L"x = \cos(x)")
plot!(xrange2, f3.(xrange2), label=L"y=\cos(x)")
r1 = (0.68, 0.82)
r2 = (0.64, 0.8)
r3 = (0.71, 0.79)
r4 = (0.69, 0.77)
scatter!(r1, label="", color="red")
scatter!(r2, label="", color="red")
scatter!(r3, label="", color="red")
scatter!(r4, label="", color="red")


# Initial guess
x0 = [0.9, 1]
x1 = newton(F, x0)
F1(x) = M(x, x1, 1, 0) * F(x)
x2 = newton(F1, x0)
F2(x) = M(x, x2) * F1(x)
x3 = newton(F2, x0)
F3(x) = M(x, x3) * F2(x)
x4 = newton(F3, x0)
F4(x) = M(x, x4) * F3(x)
x5 = newton(F4, x0)




# The Hyperbolic-Circle
# f(x, y) = xy - 1
# g(x, y) = x^2 + y^2 - 4

f1(x) = 1 / x
f2(x) = sqrt(4 - x^2)
xrange = range(-2, 2, length=100)
trange = range(0, 2π, length=100)
xrange1 = range(0.2, 3, length=100)

p = plot(xrange, f2.(xrange), color="black", label=L"x^2 + y^2 - 4=0")
plot!(xrange, -f2.(xrange), color="black", label="")
plot!(xrange1, f1.(xrange1), color="green", label=L"xy - 1 = 0")
plot!(-xrange1, -f1.(xrange1), color="green", label="")

r1 = (0.517, 1.93)
r2 = (1.93, 0.517)
r3 = (-1.93, -0.517)
r4 = (-0.517, -1.93)
scatter!(r1, label="", color="red")
scatter!(r2, label="", color="red")
scatter!(r3, label="", color="red")
scatter!(r4, label="", color="red")



F(x) = [x[1] * x[2] - 1; x[1]^2 + x[2]^2 - 4]
x0 = [-0.3, 0.3]
x1 = newton(F, x0)
F1(x) = M(x, x1) * F(x)
x2 = newton(F1, x0)
F2(x) = M(x, x2) * F1(x)
x3 = newton(F2, x0)
F3(x) = M(x, x3) * F2(x)
x4 = newton(F3, x0)
F4(x) = M(x, x4) * F3(x)
x5 = newton(F4, x0)





# Figure 1.3
f1(x) = sin(x)
p = plot(f1, -1, 8, label=L"y=\sin(x)")
f2(x) = sin(x) / x
plot!(f2, -1, 8, label=L"y=\frac{\sin(x)}{x}")
f2(x) = sin(x) / (x * (x-π))
plot!(f2, -1, 8, label=L"y=\frac{\sin(x)}{x(x-\pi)}")
hline!([0], label="", color="black")