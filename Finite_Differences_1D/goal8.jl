using ForwardDiff, LinearAlgebra, Plots
import ForwardDiff: derivative, jacobian, gradient
import LinearAlgebra: norm, inv

# First discretize the problem
n = 200
x = range(0, 10; length=n+1)
h = step(x)

# Implement the F[u] as defined in (3.9)
function F(u::AbstractVector{T}) where T # F takes a vector of lengh n+1 and returns a vector of length n+1
    v = zeros(T, length(u))
    v[2] = 1 / h^2 * (-2u[2] + u[3]) - u[2] ^ 2 + x[1]
    for k = 2: n-2
        v[k+1] = 1 / h^2 * (u[k] - 2u[k+1] + u[k+2]) - u[k+1] ^ 2 + x[k+1]
    end
    v[n] = 1 / h^2 * (u[n-1] - 2u[n] + sqrt(10)) - u[n] ^ 2 + x[n]
    v[n+1] = sqrt(10)
    v
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
        x = x - 0.5 * (qr(A) \ f(x))
        i += 1
    end
    x
end

function deflated_newton(x0, x1, f, max_iter=1000, epsilon=1e-10)
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
        x = x + 0.7 * dy
        if i > max_iter
            return "Cannot converge."
        end
        i = i + 1
    end
    x
end

x0 = zeros(n+1)
x0[end] = sqrt(10)
x1 = newton(F, x0)
x2 = newton(x0, x1, F)