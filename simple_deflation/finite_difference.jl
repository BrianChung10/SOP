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


function M(x, x1, p=2, alpha=1)
    I(n+1) / norm(x-x1)^p + alpha * I(n+1)
end


function newton(f, x0, max_iter=1000, eps=1e-10)
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
        x = x - A \ f(x)
        i += 1
    end
    x
end


function deflated_newton(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton(g, x0)
end


# We obtain the solution of the ODE via (2.6)
x0 = zeros(n+1)
x1 = newton(F, x0)
x2 = deflated_newton(x0, x1, F) # Two solutions for the ODE when lambda = 1


# Now solve the ODE via (2.10)
function deflated_newton2(x0, x1, f, max_iter=1000, eps=1e-12)
    x = x0
    i = 0
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge"
        end
        A = jacobian(f, x)
        A[1, 1] = 1
        A[end, end] = 1
        dx = -A \ f(x)
        m = M(x, x1)
        temp(y) = M(y, x1)
        m_de = jacobian(temp, x)
        dy = (I(n+1) + (I(n+1) - m \ (m_de' * dx)) \ (m \ (m_de' * dx))) * dx
        x = x + dy
        i += 1
    end
    x
end

x0 = zeros(n+1)
x1 = newton(F, x0)
x2 = deflated_newton2(x0, x1, F)

temp(x) = M(x, x1)
jacobian(temp, x0)
