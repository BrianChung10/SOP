using ForwardDiff, LinearAlgebra, Plots
import ForwardDiff: derivative, jacobian, gradient
import LinearAlgebra: norm, inv


n = 100
x = range(-1, 1; length=n + 1)
h = step(x)
ϵ = sqrt(0.00223)

# F(u) is defined as 
function F(u::AbstractVector{T}) where {T} # F takes a vector of lengh n+1 and returns a vector of length n+1
    v = zeros(T, length(u))
    v[2] = ϵ^2 / h^2 * (-2u[2] + u[3]) + 2 * (1 - x[2]^2) * u[2] + u[2]^2 - 1
    for k = 2:n-2
        v[k+1] = ϵ^2 / h^2 * (u[k] - 2u[k+1] + u[k+2]) - 2 * (1 - x[k+1]^2) * u[k+1] + u[k+1]^2 - 1
    end
    v[n] = ϵ^2 / h^2 * (u[n-1] - 2 * u[n]) - 2 * (1 - x[n]^2) * u[n] + u[n]^2 - 1
    v
end

function M(x, x1, p=2, alpha=1)
    1 / norm(x - x1)^p + alpha
end

function newton(f, x0, max_iter=1000, eps=1e-5)
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
        x = x - 0.7 * (qr(A) \ f(x))
        i += 1
    end
    x
end

x0 = zeros(n + 1)
x1 = newton(F, x0)
g1 = x -> M(x, x1) * F(x)
x2 = newton(g1, x0)
g2 = x -> M(x, x2) * g1(x)
x3 = newton(g2, x0)
g3 = x -> M(x, x3) * g2(x)
x4 = newton(g3, x0)
g4 = x -> M(x, x4) * g3(x)
x5 = newton(g4, x0)


p = plot(x, x1, label="")
p = plot(x, x2, label="")
p = plot(x, x3, label="")
plot(x, x4, lable="")