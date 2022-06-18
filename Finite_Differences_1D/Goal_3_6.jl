import Pkg
Pkg.add("ForwardDiff")
Pkg.add("LinearAlgebra")
using ForwardDiff, LinearAlgebra, Plots, LaTeXStrings
import ForwardDiff: derivative, jacobian, gradient
import LinearAlgebra: norm, inv


# First discretize the problem
n = 100
x = range(0, 1; length=n+1)
h = step(x)
lam = 1


# Implement the F[u] as defined in (3.7)
function F(u::AbstractVector{T}) where T # F takes a vector of lengh n+1 and returns a vector of length n+1
    v = zeros(T, length(u))
    v[2] = 1 / h^2 * (-2u[2] + u[3]) + lam * exp(u[2])
    for k = 2: n-2
        v[k+1] = 1 / h^2 * (u[k] - 2u[k+1] + u[k+2]) + lam * exp(u[k+1])
    end
    v[n] = 1 / h^2 * (u[n-1] - 2u[n]) + lam * exp(u[n])
    v
end


function M(x, x1, p=2, alpha=1)
    1 / norm(x-x1)^p + alpha
end


function newton(f, x0, max_iter=1000, eps=1e-10)
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


function deflated_newton(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton(g, x0)
end


# We start with an initial guess of x0 being the zero vector
x0 = zeros(n+1)
# Obtain the first solution using the Newton's method
x1 = newton(F, x0)
# Obtain the second solution using the deflation 
g = x -> M(x, x1) * F(x)
x2 = newton(g, x0)

xrange = range(0, 1, length=101)
p = plot(xrange, x1, label=L"u_1")
plot!(xrange, x2, label=L"u_2")

# We obtain the solution of the ODE via (2.10)
function deflated_newton_2(x0, x1, f, max_iter=5000, epsilon=1e-10)
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


# Define a function that compute the solutions for different lambdas
function bratu_solve(lambda, x0_1, x0_2) # x0 is the initial guess
    # Redefine the function F with different values of lambda
    function F(u::AbstractVector{T}) where T 
        v = zeros(T, length(u))
        v[2] = 1 / h^2 * (-2u[2] + u[3]) + lambda * exp(u[2])
        for k = 2: n-2
            v[k+1] = 1 / h^2 * (u[k] - 2u[k+1] + u[k+2]) + lambda * exp(u[k+1])
        end
        v[n] = 1 / h^2 * (u[n-1] - 2u[n]) + lambda * exp(u[n])
        v
    end
    x1 = newton(F, x0_1)
    x2 = deflated_newton_2(x0_2, x1, F)
    return x1, x2
end


# Compute a vector of solutions to make the plot
function solutions()
    n = 100
    solution = []
    x0_1 = zeros(n+1)
    x0_2 = zeros(n+1)
    for lambda = 0.02: 0.01: 4
        x1, x2 = bratu_solve(lambda, x0_1, x0_2)
        push!(solution, (norm(x1), norm(x2)))
        x0_2 = x2
    end
    solution
end


sol = solutions()
lambda = 0.02: 0.01: 4


line1 = [sol[i][1] for i = 1:length(sol)]
line2 = [sol[i][2] for i = 1:length(sol)]
p = plot(lambda, line1, label=L"$\|\| \mathbf{u}_1\|\|$", xlabel=L"\lambda", ylabel=L"\|\|u\|\|")
plot!(lambda, line2, label=L"$\|\| \mathbf{u}_1\|\|$")
lambda1 = (3.51383, 8.7)
scatter!(lambda1, label=L"\lambda_{*}")
vline!([lambda1[1]], label="")