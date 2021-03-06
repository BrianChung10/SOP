import Pkg
Pkg.add("DataStructures")
Pkg.add("ForwardDiff")
Pkg.add("LinearAlgebra")
using ForwardDiff, LinearAlgebra, Random
import ForwardDiff: derivative, jacobian
import LinearAlgebra: norm, inv, I


# Define the shifted deflation operator
function M(x, x1, p=2, alpha=1)
    1 / norm(x-x1)^p + alpha
end


# Define the usual newton method
function newton(f, x0, max_iter=1000, eps=1e-13)
    x = x0
    i = 0
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge."
        end
        x = x - f(x) / derivative(f, x)
        i += 1
    end
    x
end


# Define the deflated_newton in 1d
function deflated_newton(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton(g, x0)
end


function newton_higher_dimension(f, x0, max_iter=1000, eps=1e-13)
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


function deflated_newton_higher_dimension(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton_higher_dimension(g, x0)
end


function newm(x, sol, p=2, alpha=1)
    m=1
    for sols in sol
        m = m * (1 / norm(x-sols)^p + alpha)
    end
    return m
end


#Another function which finds all roots using deflated newton method
function multroot(f, x0)
    x = newton(f, x0)
    sol = [x]
    while true
        fs = y -> f(y) * newm(y,sol)
        xnew = newton(fs, x0)
        if xnew == "Cannot converge."
            return sol
        else
            push!(sol, xnew)
        end
    end
end


multroot(g, 0)


# Test functions
f(x) = (x-1) * (x-2)
g(x) = (x-2) * (x+2) * (x+3)
l(x) = (x-1)^2
m(x) = x^2-5*x+6
f1(x) = (x .- [1, 1]) .* (x .- [2, 2])


# Tests
deflated_newton(0, 1, f)
deflated_newton(0.1, 2, g)
deflated_newton(0.1, 0, sin)
deflated_newton_higher_dimension([0, 0], [1, 1], f1)


multroot(m, 0)
