using Plots, LinearAlgebra, SparseArrays


# function that computes the finite difference in 2D
function fd_2d(n)
    n -= 1
    A = Tridiagonal(fill(-1, n-1), fill(2, n), fill(-1, n-1))
    sparse(kron(A, I(n)) + kron(I(n), A))
end


n = 100
A = fd_2d(n)


n = 100
μ = 0.4
function F(u::AbstractVector{T}) where T # F takes a vector of lengh (n-1)^2 and returns a vector of length (n-1)^2
    ω = 0.2 # ω is fixed at 0.2
    A = fd_2d(n)
    xrange = range(-12, 12, length=n+1)
    yrange = range(-12, 12, length=n+1)
    h = step(xrange)

    v = 1/2 * (1 / h^2) * A * u # First term
    v += abs.(u) .^ 2 .* u # Second term
    index = 1
    for i = 2: length(xrange)-1  # Third term
        for j = 2: length(yrange)-1
            v[index] += 1 / 2 * ω ^ 2 * (xrange[i] ^ 2 + yrange[j] ^ 2) * u[index]
            index += 1
        end
    end
    v -= μ * u
    v
end


function M(x, x1, p=4, alpha=1) # Modified deflation operator
    1 / norm(x-x1)^p + alpha
end


function newton(f, x0, max_iter=1000, eps=1e-4)
    x = x0
    i = 0
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge."
        end
        A = jacobian(f, x)
        x = x - (qr(A) \ f(x)) # damped
        i += 1
    end
    x
end

function deflated_newton(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton(g, x0)
end


# We obtain the solution of the ODE via (2.6)
x0 = zeros((n-1)^2)
x1 = newton(F, x0) 
x2 = deflated_newton(x0, x1, F) # Two solutions for the ODE
