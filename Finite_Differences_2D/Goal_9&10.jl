# Goal 9 Matrix
using PlotlyJS

# function that computes the finite difference in 2D
function fd_2d(n)
    n -= 1
    A = Tridiagonal(fill(-1, n-1), fill(2, n), fill(-1, n-1))
    kron(A, I(n)) + kron(I(n), A)
end

fd_2d(4)
fd_2d(5)

# Approximately solve Possion's equation with homogeneous Dirichlet boundary conditions
function possion_solve(n)
    xrange = range(0, 1, length=n+1)
    yrange = range(0, 1, length=n+1)
    h = step(xrange)

    f(x, y) = 2π^2 * sin(π*x)cos(π*y)

    f_vec = zeros((n-1)^2)
    index = 1
    for i = 2: 100
        for j = 2: 100
            f_vec[index] = f(xrange[i], yrange[j])
            index += 1
        end
    end
    f_vec
    A = (1 / h^2) * fd_2d(n)
    A \ f_vec
end

sol = possion_solve(100)

plot(contour(A))