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


# Solve for all roots by using deflated newton method
function deflated_newton_solve_1d(f, x0)
    x = newton(f, x0)
    solution = [x]
    while x != "Cannot converge."
        f = y -> f(y) * M(y, x)
        x0 = rand(-1000: 1000)
        x = newton(f, x0)
        push!(solution, x)
    end
    solution
end


g(x) = (x-2) * (x+2) * (x+3)
fs = Vector{Function}(undef, 1)
fs[1] = f
push!(fs, g)
i(x) = last(fs)(x) * (x-1)
i(5)


function multroot(f, x0, n = 10)
    fs = Vector{Function}(undef, n)
    fs[1] = f
    xn = newton(f, 0)
    sol = [xn]
    k = 1
    while xn != "Cannot converge."
        k += 1
        xn = last(sol)
        fs[k] = x -> fs[k-1](x) * M(x, xn)
        xnew = newton(fs[k], x0)
        push!(sol, xnew)
    end
    sol
end

multroot(g, 0)



fs = Vector{Function}(undef, 10)
fs[1] = g
sol = [newton(g,0)]
newton(g,0)
fs[2] = x -> fs[1](x) * M(x, -3)
newton(fs[2], 0)
fs[3] = x -> fs[2](x) * M(x, -2)
newton(fs[3], 0)
fs[4] = x -> fs[3](x) * M(x, 2)
newton(fs[4], 0)


sol = [x1]
xn = last(sol)
fun(x) = last(fs)(x) * M(x, xn)
newton(fun, 0)
fun1(x) = fun(x) * M(x, -2)
newton(fun1, 0)
fun2(x) = fun1(x) * M(x, 2)
newton(fun2, 0)



# Test functions
f(x) = (x-1) * (x-2)
g(x) = (x-2) * (x+2) * (x+3)
h(x) = sin(x)
f1(x) = (x .- [1, 1]) .* (x .- [2, 2])

# Tests
deflated_newton(0, 1, f)
deflated_newton(0.1, 2, g)
deflated_newton(0.1, 0, h)
deflated_newton_higher_dimension([0, 0], [1, 1], f1)
deflated_newton_solve_1d(f, 0)

mult_root(f, 1.2, 10)