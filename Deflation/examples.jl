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


x0 = [0.5; 0.5] # Initial guess
h1(x) = [f(x[1], x[2]); g(x[1], x[2])] # Original function (Cubic-Parabola)
x1 = newton(h1, x0) # Find the first root
h2(x) = M(x, x1) * h1(x) # Define the deflated function
x2 = newton(h2, x0) # Find the second root with the same intial guess
h3(x) = M(x, x2) * h2(x) # Define the second deflated function
x3 = newton(h3, x0)
sol = [x1, x2, x3]

struct DeflatedFunction{F}
    x::Vector{Float64} # points
    f::F # original function
end

function (D::DeflatedFunction)(x)
    ret = D.f(x)
    for x in D.x
        # TODO: update ret
    end
    ret
end



# Example 2: The Four-Cluster
f(x, y) = (x - y^2) * (x - sin(y))
g(x, y) = (cos(y) - x) * (y - cos(x))


function four_cluster(x)
    [f(x[1], x[2]); g(x[1], x[2])]
end


x0 = [0.9; 1]
newton(four_cluster, x0)
