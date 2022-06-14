using PlotlyJS, LinearAlgebra, SparseArrays
import ForwardDiff: derivative, jacobian, gradient
import LinearAlgebra: norm, inv
import SparseArrays: sparse

# function that computes the finite difference in 2D
function fd_2d(n)
    n -= 1
    A = Tridiagonal(fill(-1, n-1), fill(2, n), fill(-1, n-1))
    kron(sparse(A), sparse(I(n))) + kron(sparse(I(n)), sparse(A))
end

A = fd_2d(4)


n = 100
μ = 0.8 # Now make μ = 4ω
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
    1 / norm(x - x1)^p + alpha
end


function jacobian_s(u) # u is a (n-1) * (n-1) vector
    ω = 0.2
    v = zeros((n-1)^2)

    xrange = range(-12, 12, length=n+1)
    yrange = range(-12, 12, length=n+1)
    h = step(xrange)

    index = 1
    for i = 2: n
        for j = 2: n
            v[index] += 1 / 2 * ω^2 * (xrange[i]^2 + yrange[j]^2)
            index += 1
        end
    end
    v = v .- μ
    v += 3 * u .^ 2
    (1/2) * (1/h^2) * fd_2d(n) + Diagonal(v)
end

x0 = zeros((n-1)^2)
A = jacobian_s(x0)
B = jacobian(F, x0)
A - B


function newton(f, x0, max_iter=1000, eps=1e-5)
    x = x0
    i = 0
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge."
        end
        A = sparse(jacobian(f, x))
        x = x - 0.7 * (qr(A) \ f(x)) 
        i += 1
    end
    x
end


function newton_s(f, x0, max_iter=1000, eps=1e-5)
    x = x0
    i = 0
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge."
        end
        A = jacobian_s(x)
        x = x - 0.6 * (qr(A) \ f(x)) 
        i += 1
    end
    x
end


function deflated_newton(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton(g, x0)
end


# Newton with hand coded jacobian matrix
x0 = -0.05 .* ones((n-1)^2)
x1 = newton_s(F, x0)
F1(x) = M(x, x1) * F(x)
x2 = newton_s(F1, x0)
F2(x) = M(x, x2) * F1(x)
x3 = newton_s(F2, x0)
F3(x) = M(x, x3) * F2(x)
x4 = newton_s(F3, x0)
F4(x) = M(x, x4) * F3(x)
x5 = newton_s(F4, x0)
F5(x) = M(x, x5) * F4(x)
x6 = newton_s(F5, x0)
F6(x) = M(x, x6) * F5(x)
x7 = newton_s(F6, x0)
F7(x) = M(x, x7) * F6(x)
x8 = newton_s(F7, x0)
F8(x) = M(x, x8) * F7(x)
x9 = newton_s(F8, x0)







x0 = 1/2 .* ones((n-1)^2)
x1 = newton(F, x0)
F1(x) = M(x, x1) * F(x)
x2 = newton(F1, x0)
F2(x) = M(x, x2) * F1(x)
x3 = newton(F2, x0)
F3(x) = M(x, x3) * F2(x)
x4 = newton(F3, x0)
F4(x) = M(x, x4) * F3(x)
x5 = newton(F4, x0)
F5(x) = M(x, x5) * F4(x)
x6 = newton(F5, x0)
F6(x) = M(x, x6) * F5(x)
x7 = newton(F6, x0)
F7(x) = M(x, x7) * F6(x)
x8 = newton(F7, x0)


x1_mat = reshape(x1, n-1, n-1)
x2_mat = reshape(x2, n-1, n-1)
x3_mat = reshape(x3, n-1, n-1)
x4_mat = reshape(x4, n-1, n-1)
x5_mat = reshape(x5, n-1, n-1)
x6_mat = reshape(x6, n-1, n-1)
x7_mat = reshape(x7, n-1, n-1)
x8_mat = reshape(x8, n-1, n-1)
x9_mat = reshape(x9, n-1, n-1)


xrange = [i for i = range(-12, 12, length=n+1)]
yrange = [i for i = range(-12, 12, length=n+1)]

data1 = contour(x=xrange, y=yrange, z=x1_mat, contours_coloring="heatmap", line_width=0)
plot(data1)

data2 = contour(x=xrange, y=yrange, z=x2_mat, contours_coloring="heatmap", line_width=0)
plot(data2)

data3 = contour(x=xrange, y=yrange, z=x3_mat, contours_coloring="heatmap", line_width=0)
plot(data3)

data4 = contour(x=xrange, y=yrange, z=x4_mat, contours_coloring="heatmap", line_width=0)
plot(data4)

data5 = contour(x=xrange, y=yrange, z=x5_mat, contours_coloring="heatmap", line_width=0)
plot(data5)

data6 = contour(x=xrange, y=yrange, z=x6_mat, contours_coloring="heatmap", line_width=0)
plot(data6)

data7 = contour(x=xrange, y=yrange, z=x7_mat, contours_coloring="heatmap", line_width=0)
plot(data7)

data8 = contour(x=xrange, y=yrange, z=x8_mat, contours_coloring="heatmap", line_width=0)
plot(data8)

data9 = contour(x=xrange, y=yrange, z=x9_mat, contours_coloring="heatmap", line_width=0)
plot(data9)