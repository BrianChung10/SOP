using ForwardDiff, LinearAlgebra, Plots
import ForwardDiff: derivative, jacobian, gradient
import LinearAlgebra: norm, inv, cond


# First discretize the problem
n = 100
x = range(0, 1; length=n+1)
h = step(x)
lambda = 1

# Implement the F[u] as defined in (3.7)
function F(u::AbstractVector{T}) where T # F takes a vector of lengh n+1 and returns a vector of length n+1
    v = zeros(T, length(u))
    v[2] = 1 / h^2 * (-2u[2] + u[3]) + lambda * exp(u[2])
    for k = 2: n-2
        v[k+1] = 1 / h^2 * (u[k] - 2u[k+1] + u[k+2]) + lambda * exp(u[k+1])
    end
    v[n] = 1 / h^2 * (u[n-1] - 2u[n]) + lambda * exp(u[n])
    v
end


# Implement the shifted deflation operator
function M(x, x1, p=2, alpha=1)
    1 / norm(x-x1)^p + alpha
end


# Modify the orginal newton method to return the condition number of the jacobian after each iteration
function newton_cond(f, x0, max_iter=1000, eps=1e-10)
    x = x0
    i = 0
    cond_number = []
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge"
        end
        A = jacobian(f, x)
        # Have to make A invertible while preserving the boundary conditions
        A[1, 1] = 1
        A[end, end] = 1
        push!(cond_number, cond(A))
        x = x - 0.6 * A \ f(x) # Damping newton iteration
        i += 1
    end
    cond_number
end


function deflated_newton_cond(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton_cond(g, x0)
end


# We obtain the evolution of the condition number via (2.7)

cond_number = newton_cond(F, x0)
cond_number_deflated = deflated_newton_cond(x0, x1, F)
plot(log.(cond_number), title= "Evolution of the condition number 
(naive implementations)", label="First solution")
plot!(log.(cond_number_deflated), label="Second solution")


# We obtain the solution of the ODE via (2.10)
function deflated_newton_cond2(x0, x1, f, max_iter=1000, epsilon=1e-10, p=2)
    x = x0
    i = 0
    cond_number = []
    while norm(f(x)) > epsilon
        B = jacobian(f, x)
        B[1, 1] = 1
        B[end, end] = 1
        push!(cond_number, cond(B))
        dx = -B \ f(x)
        m = M(x,x1)
        temp_func(y) = M(y, x1)
        m_de = gradient(temp_func, x)
        dy = (1+(1 / m)*m_de'*dx/(1-(1 / m)*m_de'*dx))*dx
        x = x + 0.6 * dy # Damping newton iteration
        if i > max_iter
            return "Cannot converge."
        end
        i = i + 1
    end
    cond_number
end

cond_number2 =  deflated_newton_cond2(x0, x1, F)
plot(log.(cond_number), title= "Evolution of the condition number 
(naive implementations)", label="First solution")
plot!(log.(cond_number2), label="Second solution")
