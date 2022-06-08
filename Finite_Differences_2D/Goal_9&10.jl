import Pkg
Pkg.add("PlotlyJS")
using PlotlyJS
using LinearAlgebra
using SparseArrays
import LinearAlgebra: norm


# function that computes the finite difference in 2D
function fd_2d(n)
    n -= 1
    A = Tridiagonal(fill(-1, n-1), fill(2, n), fill(-1, n-1))
    sparse(kron(A, I(n)) + kron(I(n), A))
end


fd_2d(4)
fd_2d(5)


# Approximately solve Possion's equation with homogeneous Dirichlet boundary conditions
function possion_solve(n)
    xrange = range(0, 1, length=n+1)
    yrange = range(0, 1, length=n+1)
    h = step(xrange)

    f(x, y) = 2π^2 * sin(π * x) * sin(π * y)

    f_vec = zeros((n-1)^2)
    index = 1
    for i = 2: 100
        for j = 2: 100
            f_vec[index] = f(xrange[i], yrange[j])
            index += 1
        end
    end

    A = (1 / h^2) * fd_2d(n)
    A \ f_vec
end


qsol = possion_solve(100)


function poisson_contour(n)
    sol = possion_solve(n)
    sol = reshape(sol, n-1, n-1)

    data = contour(; z=sol)
    layout = Layout(;title="Contour Plot of Poisson Equation")
    plot(data, layout)
end


poisson_contour(100)


# Here is the solution of the Poisson equation in this case
u(x, y) = sin(π * x) * sin(π * y)


n = 100
xrange = range(0, 1, length=n+1)
yrange = range(0, 1, length=n+1)
u_vec = zeros((n-1)^2)
    index = 1
    for i = 2: n
        for j = 2: n
            u_vec[index] = u(xrange[i], yrange[j])
            index += 1
        end
    end
u_vec


u_matrix = reshape(u_vec, n-1, n-1)


data = contour(; z=u_matrix)
layout = Layout(;title="Contour Plot of Poisson Equation")
plot(data, layout)


# Find the different numerical and actual solution
error = norm(u_vec - sol)


function Schrodinger(n, omega, mu, u::AbstractVector{T}) where T
    xrange = range(-12, 12, length = n+1)
    yrange = range(-12, 12, length = n+1)
    h = 1/n

    f_vec = zeros((n-1)^2)
    f(x, y) = norm(u)^2 + 1/2 * omega^2 * (x^2+y^2) - mu
    index = 1
    for i = 2: 100
        for j = 2: 100
            f_vec[index] = f(xrange[i], yrange[j]) * u[index]
            index += 1
        end
    end
    v = 1/(2*h^2) * fd_2d(n) * u + f_vec
    v
end

