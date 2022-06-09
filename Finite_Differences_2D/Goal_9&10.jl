using PlotlyJS
using Plots, LaTeXStrings
using LinearAlgebra
using SparseArrays

# function that computes the finite difference in 2D
function fd_2d(n)
    n -= 1
    A = Tridiagonal(fill(-1, n-1), fill(2, n), fill(-1, n-1))
    kron(sparse(A), sparse(I(n))) + kron(sparse(I(n)), sparse(A))
end

A = fd_2d(4)


# Approximately solve Possion's equation with homogeneous Dirichlet boundary conditions
function poisson_solve(n)
    xrange = range(0, 1, length=n+1)
    yrange = range(0, 1, length=n+1)
    h = step(xrange)

    f(x, y) = 2π^2 * sin(π * x) * sin(π * y)

    f_vec = zeros((n-1)^2)
    index = 1
    for i = 2: n
        for j = 2: n
            f_vec[index] = f(xrange[i], yrange[j])
            index += 1
        end
    end

    A = (1 / h^2) * fd_2d(n)
    A \ f_vec
end

sol = poisson_solve(100)

# Change color of the contour
colorscale = [[0, "black"], [1, "white"]]

function poisson_contour(n)
    sol = poisson_solve(n)
    sol = reshape(sol, n-1, n-1)

    data = contour(; z=sol, contours_coloring="heatmap", line_width=0)
    layout = Layout(;title="Contour Plot of Poisson Equation")
    plot(data, layout)
end

poisson_contour(100)

# Here is the solution of the Poisson equation in this case
u(x, y) = sin(π * x) * sin(π * y)

n = 100
function actuaL_sol(n)
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
end

u_vec = actuaL_sol(n)

u_matrix = reshape(u_vec, n-1, n-1)

data = contour(; z=u_matrix)
layout = Layout(;title="Contour Plot of Poisson Equation")
plot(data, layout)

# Find the different numerical and actual solution
error = norm(u_vec - sol)

sol = poisson_solve(20)
u_vec = actuaL_sol(20)

function error_approx(up_bound)
    error = []
    for n = 10: up_bound
        sol = poisson_solve(n)
        u_vec = actuaL_sol(n)
        push!(error, norm(sol-u_vec))
    end
    error
end

error = error_approx(200)
error1 = float.(error)
n = 10: 200

plot(n, error1, title="Difference between numerical and actual solution", label="Error")