import Pkg
Pkg.add("ForwardDiff")
Pkg.add("LinearAlgebra")
using ForwardDiff
using LinearAlgebra
import ForwardDiff: derivative, jacobian
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


f(x) = (x-1) * (x+1)
g(x) = (x-2) * (x+2) * (x+3)
h(x) = sin(x)


deflated_newton(-0.1, 1, f)
deflated_newton(0.1, 2, g)
deflated_newton(0.1, 0, h)
