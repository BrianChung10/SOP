import Pkg
Pkg.add("ForwardDiff")
Pkg.add("LinearAlgebra")
Pkg.add("Plots")
using ForwardDiff, LinearAlgebra, Plots, LaTeXStrings
import ForwardDiff: derivative, jacobian, gradient
import LinearAlgebra: norm, inv


# First discretize the problem
n = 100
x = range(0, 10; length=n+1)
h = step(x)


# Implement the F[u] as defined in (3.7)
function F(u::AbstractVector{T}) where T # F takes a vector of lengh n+1 and returns a vector of length n+1
    v = zeros(T, length(u))
    v[2] = 1 / h^2 * (-2u[2] + u[3]) - (u[2])^2 + x[2]
    for k = 2: n-2
        v[k+1] = 1 / h^2 * (u[k] - 2u[k+1] + u[k+2]) - (u[k+1])^2 + x[k+1]
    end
    v[n] = 1 / h^2 * (u[n-1] - 2u[n] + sqrt(10)) - u[n] ^ 2 + x[n]
    v[n+1] = u[n+1] - sqrt(10) # boundary condition not 0
    v
end


function M(x, x1, p=2, alpha=1)
    1 / norm(x-x1)^p + alpha
end


function newton(f, x0, max_iter=1000, eps=1e-4)
    x = x0
    i = 0
    sol = []
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge."
        end
        A = jacobian(f, x)
        # Have to make A invertible while preserving the boundary conditions
        A[1, 1] = 1
        A[end, end] = 1
        x = x - (qr(A) \ f(x)) # damped
        push!(sol, x)
        i += 1
    end
    sol
end



function deflated_newton(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton(g, x0)
end



# We obtain the solution of the ODE via (2.6)
x0 = zeros(n+1)
x0[end] = sqrt(10)

sol = newton(F, x0)
plot(sol, label="")

x1 = newton(F, x0)
x2 = deflated_newton(x0, x1, F) # Two solutions for the ODE

xrange = range(0, 10, length=n+1)

p = plot(xrange, x1, legend=:bottomright, label=L"u_1")
plot!(xrange, x2, label=L"u_2")

# We obtain the solution of the ODE via (2.10)
function deflated_newton_2(x0, x1, f, max_iter=1000, epsilon=1e-8)
    x = x0
    i = 0
    while norm(f(x)) > epsilon
        B = jacobian(f, x)
        B[1, 1] = 1
        B[end, end] = 1
        dx = -B \ f(x)
        m = M(x,x1)
        temp_func(y) = M(y, x1)
        m_de = gradient(temp_func, x)
        dy = (1+(1 / m)*m_de'*dx/(1-(1 / m)*m_de'*dx))*dx
        x = x + dy
        if i > max_iter
            return "Cannot converge."
        end
        i = i + 1
    end
    x
end

x3 =  deflated_newton_2(x0, x1, F)
plot(x3)
# tried p = 3, p = 4 and a = 0, damped factor 0.9 -- simple vs standard gives different sols