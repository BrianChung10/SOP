using ForwardDiff, LinearAlgebra
import LinearAlgebra: inv
import ForwardDiff: jacobian


# First discretize the problem
n = 100
x = range(0, 1; length=n+1)
h = step(x)
lambda = 1


# Implement the F[u] as defined in (3.7)
function F(u::AbstractVector{T}) where T # F takes a vector of lengh n+1 and returns a vector of length n+1
    v = zeros(T, length(u))
    v[2] = 1 / h^2 * (-2u[2] + u[3]) + lambda * exp(u[2])
    for k = 2: n-2
        v[k+1] = 1 / h^2 * (u[k] - 2u[k+1] + u[k+2]) + lambda * exp(u[k+1])
    end
    v[n] = 1 / h^2 * (u[n-1] - 2u[n]) + lambda * exp(u[n])
    v
end


y = zeros(n+1)
A = jacobian(F, y)


# Make jacobian invertible while preserving the boundary conditions
A[1, 1] = 1
A[end, end] = 1

function M(x, x1, p=2, alpha=1)
    I(n+1) / norm(x-x1)^p + alpha * I(n+1)
end


function newton(f, x0, max_iter=1000, eps=1e-13)
    x = x0
    i = 0
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge"
        end
        A = jacobian(f, x)
        # Have to make A invertible while preserving the boundary conditions
        A[1, 1] = 1
        A[end, end] = 1
        x = x - inv(A) * f(x)
        i += 1
    end
    x
end


function deflated_newton(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton(g, x0)
end


x0 = zeros(n+1)
x1 = newton(F, x0)
