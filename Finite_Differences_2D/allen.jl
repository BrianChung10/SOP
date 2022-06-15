using PlotlyJS, LinearAlgebra
import ForwardDiff: derivative, jacobian, gradient
import LinearAlgebra: norm, inv
import SparseArrays: sparse

n = 50

# This function helps us to produce the initial guess
function make_U(n)
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

function M(x, x1, p=2, alpha=1) # Modified deflation operator
    1 / norm(x - x1)^p + alpha
end

δ = 0.04
# Define function F in this case
function F(u::AbstractVector{T}) where T
    # F takes a vector of length (n+1)^2 and returns a vector of (n+1)^2
    U = reshape(u, (n+1, n+1))'
    v = zeros(T, n+1, n+1) # Make a matrix first and then convert it to a vector

    # Need to satisfy the boundary condition
    for i = 1: n+1
        for j = 1:n+1
            if (j == 1 || j == n+1) && (i != 1 && i != n+1)
                v[i, j] = 1 - U[i, j]
            elseif i == 1 || i == n+1
                v[i, j] = -1 - U[i, j]
            end
        end
    end

    xrange = range(0, 1, length=n+1)
    yrange = range(0, 1, length=n+1)
    h = step(xrange)

    for i = 2: n
        for j = 2: n
            v[i, j] = -δ * (1 / h^2) * (-4U[i, j] + U[i, j-1] + U[i, j+1] + U[i-1, j] + U[i+1, j]) + 1/δ * (U[i, j]^3 - U[i, j])
        end
    end
    v = vec(v')
end


function newton(f, x0, max_iter=1000, eps=1e-5)
    x = x0
    i = 0
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge"
        end
        A = jacobian(f, x)
        for i = 1: n+1
            for j = 1: n+1
                if i == j && A[i, j] == 0
                  A[i, j] = 1
                end
            end
        end
        x = x - 0.1 * (qr(A) \ f(x))
        i += 1
    end
    x
end

U = make_U(n)
U[2:end-1, 2:end-1] .= 0.1
x0 = vec(U') # Initial guess that satisfies the boundary condition

x1 = newton(F, x0)
F1(x) = M(x, x1) * F(x)
x2 = newton(F1, x0)
F2(x) = M(x, x2) * F1(x)
x3 = newton(F2, x0)
F3(x) = M(x, x3) * F2(x)
x4 = newton(F3, x0)


x1_mat = reshape(x1, n+1, n+1)
x2_mat = reshape(x2, n+1, n+1)
x3_mat = reshape(x3, n+1, n+1)


xrange = [i for i = range(0, 1, length=n+1)]
yrange = [i for i = range(0, 1, length=n+1)]

data1 = contour(x=xrange, y=yrange, z=x1_mat, contours_coloring="heatmap", line_width=0)
plot(data1)

data2 = contour(x=xrange, y=yrange, z=x2_mat, contours_coloring="heatmap", line_width=0)
plot(data2)

data3 = contour(x=xrange, y=yrange, z=x3_mat, contours_coloring="heatmap", line_width=0)
plot(data3)