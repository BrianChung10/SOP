import Pkg
Pkg.add("ForwardDiff")
Pkg.add("LinearAlgebra")
using ForwardDiff
using LinearAlgebra
import ForwardDiff: derivative, jacobian, gradient
import LinearAlgebra: norm


# Define the shifted deflation operator
function M(x, x1, p=2, alpha=1)
    1 / norm(x-x1)^p + alpha
end


# Implement the deflated_newton in one dimension
function deflated_newton(x0, x1, f, max_iter=1000, epsilon=1e-13, p=2)
    x = x0
    i = 0
    while abs(f(x)) > epsilon
        dx = - ((derivative(f, x))^(-1))*f(x)
        m = M(x,x1)
        m_de = (-p * (x-x1) / norm(x-x1)^(p+2))
        dy = (1+m^(-1)*m_de*dx/(1-m^(-1)*m_de*dx))*dx
        x = x + dy 
        if i > max_iter
            return "Cannot converge."
        end
        i = i + 1
    end
    x
end


# Implement the deflated_newton in higher dimension
function deflated_newton_higher_dimension(x0, x1, f, max_iter=1000, epsilon=1e-13, p=2)
    x = x0
    i = 0
    while norm(f(x)) > epsilon
        dx = -inv(jacobian(f,x))*f(x)
        m = M(x,x1)
        temp_func(y) = M(y, x1)
        m_de = gradient(temp_func, x)
        dy = (1+inv(m)*m_de'*dx/(1-inv(m)*m_de'*dx))*dx
        x = x + dy
        if i > max_iter
            return "Cannot converge."
        end
        i = i + 1
    end
    x
end


function M1(x, sol, p=2, alpha=1)
    m=1
    for sols in sol
        m = m * (1 / norm(x-sols)^p + alpha)
    end
    return m
end


# Solve for all roots by using deflated newton method
function deflated_newton_solve_1d(x0, x1, f)
    x = deflated_newton(x0, x1, f)
    solution = [x]
    fs = x -> f(x)*M(x,x1)
    while true
        x = deflated_newton(x0, x, fs)
        fs = y -> f(y) * M1(y,solution)
        if x == "Cannot converge."
            return solution
        else
            push!(solution, x)
        end
    end
end


function deflated_newton_solve_nd(x0, x1, f)
    x = deflated_newton_higher_dimension(x0, x1, f)
    solution = [x]
    fs = x -> f(x)*M(x,x1)
    while true
        x = deflated_newton_higher_dimension(x0, x, fs)
        fs = y -> f(y) * M1(y,solution)
        if x == "Cannot converge."
            return solution
        else
            push!(solution, x)
        end
    end
end


# Test functions
f(x) = (x-1) * (x+1)
g(x) = (x-2) * (x+2) * (x+3)
h(x) = sin(x)
f1(x) = (x .- [1, 1]) .* (x .- [2, 2])
f2(x) = (x .- [1, 1]) .* (x .- [2, 2]).*(x .- [3, 3])
z(x) = (x-1) * (x-2) * (x-3) * (x-4)


# Tests
deflated_newton(-0.1, 1, f)
deflated_newton(0.1, 2, g)
deflated_newton(0.1, 0, sin)
deflated_newton_higher_dimension([0.1,0.1],[1, 1],f1)
deflated_newton_solve_1d(1, 2, g)
deflated_newton_solve_1d(0, 1, z)
deflated_newton_solve_nd([0.1,0.1],[1, 1],f2)
deflated_newton_solve_nd([0.1,0.1],[1, 1],f1)