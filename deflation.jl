import Pkg
Pkg.add("ForwardDiff")

 

using ForwardDiff

 

function M(x, x1, p=2, alpha=1)
    1 / abs(x-x1)^p + alpha
end

function deflated_newton(x0, x1, f, max_iter=1000, epsilon=1e-10, p=2)
    x = x0
    i = 0
    while abs(f(x)) > epsilon
        x = x - 1 / ((-p * (x-x1) / abs(x-x1)^(p+2)) * f(x) +
            M(x, x1) *ForwardDiff.derivative(f, x)) * M(x, x1) * f(x)
        if i > max_iter
            return "Cannot converge."
        end
        i = i + 1
    end
    x
end

 

function test_deflated_newton(f, x0, x1)
    deflated_newton(x0, x1, f)
end

 

f(x) = x^2 - 1

deflated_newton(-0.1, 1, f)

 