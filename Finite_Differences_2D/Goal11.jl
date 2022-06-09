using PlotlyJS, LinearAlgebra, SparseArrays
import ForwardDiff: derivative, jacobian, gradient
import LinearAlgebra: norm, inv

# function that computes the finite difference in 2D
function fd_2d(n)
    n -= 1
    A = Tridiagonal(fill(-1, n-1), fill(2, n), fill(-1, n-1))
    kron(A, I(n)) + kron(I(n), A)
end


n = 4
A = fd_2d(n)


n = 50
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


function M(x, x1, p=2, alpha=1) # Modified deflation operator
    1 / norm(x-x1)^p + alpha
end


function newton(f, x0, max_iter=1000, eps=1e-5)
    x = x0
    i = 0
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge."
        end
        A = sparse(jacobian(f, x))
        x = x - (qr(A) \ f(x)) 
        i += 1
    end
    x
end

function deflated_newton(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton(g, x0)
end


x0 = 1/2 .* ones((n-1)^2)
x1 = newton(F, x0)
F1(x) = M(x, x1) * F(x)
x2 = newton(F1, x0)
F2(x) = M(x, x2) * F1(x)
x3 = newton(F2, x0)
F3(x) = M(x, x3) * F2(x)
x4 = newton(F3, x0)

x1_mat = reshape(x1, n-1, n-1)
x2_mat = reshape(x2, n-1, n-1)
x3_mat = reshape(x3, n-1, n-1)

data1 = contour(; z=x1_mat, contours_coloring="heatmap", line_width=0)
plot(data1)

data2 = contour(; z=x2_mat, contours_coloring="heatmap", line_width=0)
plot(data2)

data3 = contour(; z=x3_mat, contours_coloring="heatmap", line_width=0)
plot(data3)