using ForwardDiff
import ForwardDiff: jacobian


# First discretize the problem
n = 100
x = range(0, 1; length=n+1)
h = step(x)
lambda = 1


# Implement the F[u] as defined in (3.7)
function F(u) # F takes a vector of lengh n+1 and returns a vector of length n+1
    v = zeros(length(u))
    for k = 1: n-2
        v[k+1] = 1 / h^2 * (u[k] - 2u[k+1] + u[k+2]) + lambda * exp(u[k+1])
    end
    v
end

y = zeros(n+1)
jacobian(F, y)

F(y)
