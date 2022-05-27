import Pkg
Pkg.add("ForwardDiff")
Pkg.add("LinearAlgebra")
using ForwardDiff
using LinearAlgebra
import ForwardDiff: derivative, jacobian
import LinearAlgebra: norm, inv


# Define the shifted deflation operator
function M(x, x1, p=2, alpha=1)
    1 / norm(x-x1)^p + alpha
end


# Implement the deflated_newton in one dimension
function deflated_newton(x0, x1, f, max_iter=1000, epsilon=1e-13, p=2)
    x = x0
    i = 0
    while abs(f(x)) > epsilon
        x = x - 1 / ((-p * (x-x1) / norm(x-x1)^(p+2)) * f(x) +
            M(x, x1) * derivative(f, x)) * M(x, x1) * f(x)
        if i > max_iter
            return "Cannot converge."
        end
        i = i + 1
    end
    x
end


function deflated_newton_higher_dimension(x0, x1, f, max_iter=1000, epsilon=1e-13, p=2)
    x = x0
    i = 0
    while norm(f(x)) > epsilon
        temp_func(y) = M(y, x1) * f(y)
        x = x - inv(jacobian(temp_func, x)) * M(x, x1) * f(x)
        if i > max_iter
            return "Cannot converge"
        end
        i = i + 1
    end
    x
end

f(x) = (x-1) * (x+1)
g(x) = (x-2) * (x+2) * (x+3)
h(x) = sin(x)
f1(x) = (x .- [1, 1]) .* (x .- [2, 2])


deflated_newton(-0.1, 1, f)
deflated_newton(0.1, 2, g)
deflated_newton(0.1, 0, h)
