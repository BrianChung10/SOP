using PlotlyJS, LinearAlgebra
import ForwardDiff: derivative, jacobian, gradient
import LinearAlgebra: norm, inv
import SparseArrays: sparse

n = 100
function make_U(n) # Function that produces the matrix satisfies the boundary condition
    xrange = range(0, 1, length=n+1)
    yrange = range(0, 1, length=n+1)
    h = step(xrange)

    A = zeros(n+1, n+1)
    # Need to satisfy the boundary condition
    for i = 1: n+1
        for j = 1:n+1
            if (j == 1 || j == n+1) && (i != 1 && i != n+1)
                A[i, j] = 1
            elseif i == 1 || i == n+1
                A[i, j] = -1
            end
        end
    end
A
end

A = make_U(10)



δ = 0.04
# Define function F in this case
function F(u::AbstractVector{T}) where T
    # F takes a vector of length (n+1)^2 and returns a vector of (n+1)^2
    U = reshape(u, (n+1, n+1))'
    v = make_U(n)

    xrange = range(0, 1, length=n+1)
    yrange = range(0, 1, length=n+1)
    h = step(xrange)

    for i = 2: n
        for j = 2: n
            v[i, j] = -δ * (4U[i, j] + U[i, j-1] + U[i, j+1] + U[i-1, j] + U[i+1, j]) - h^2 * 1/δ * (U[i, j]^3 - U[i, j])
        end
    end

    convert(Vector{Float64}, vec(v'))
end


function newton(f, x0, max_iter=1000, eps=1e-8)
    x = x0
    i = 0
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge"
        end
        A = jacobian(f, x)
        x = x - (qr(A) \ f(x))
        i += 1
    end
    x
end


x1 = newton(F, x0)
