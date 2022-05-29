import Pkg
Pkg.add("Plots")
using Plots
using ForwardDiff, LinearAlgebra
import ForwardDiff: derivative, jacobian
import LinearAlgebra: norm, inv, I

function M(x, x1, p=2, alpha=1)
    I(2) / norm(x-x1)^p + alpha * I(2)
end


function newton(f, x0, max_iter=1000, eps=1e-13)
    x = x0
    i = 0
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge."
        end
        x = x - inv(jacobian(f, x)) * f(x)
        i += 1
    end
    x
end


function deflated_newton(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton(g, x0)
end


# Example 1: The Cubic-Parabola
f(x, y) = 4x^3 - 3x - y
g(x, y) = x^2 - y

f1(x) = 4x^3 - 3x
g1(x) = x^2

plot(f1, -1.5, 1.5)
plot!(g1, -1.5, 1.5)

function cubic_parabola(x)
    [f(x[1], x[2]); g(x[1], x[2])]
end

x0 = [0.5; 0.5]
newton(cubic_parabola, x0)

x0 = [-1; -1]
x1 = newton(cubic_parabola, x0)
x2 = deflated_newton(x0, x1, cubic_parabola)
