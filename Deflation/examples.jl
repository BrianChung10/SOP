using Plots
using ForwardDiff, LinearAlgebra, LaTeXStrings
import ForwardDiff: derivative, jacobian
import LinearAlgebra: norm, inv, I


function M(x, x1, p=2, alpha=1)
    1 / norm(x-x1)^p + alpha
end


function newton(f, x0, max_iter=1000, eps=1e-13)
    x = x0
    i = 0
    while norm(f(x)) > eps
        if i > max_iter
            return "Cannot converge."
        end
        x = x - inv(jacobian(f, x)) * f(x)
        i += 1
    end
    x
end


function deflated_newton(x0, x1, f)
    g = x -> M(x, x1) * f(x)
    newton(g, x0)
end


# Example 1: The Cubic-Parabola
f(x, y) = 5x^3 - 4x - y
g(x, y) = 3x^2 + x - y

f1(x) = 5x^3 - 4x
g1(x) = 3x^2 + x

p = plot(f1, -1.5, 1.5, label=L"5x^3-4x-y=0")
plot!(g1, -1.5, 1.5, label=L"3x^2+x-y=0")


x0 = [0.5; 0.5] # Initial guess
h1(x) = [f(x[1], x[2]); g(x[1], x[2])] # Original function (Cubic-Parabola)
x1 = newton(h1, x0) # Find the first root
h2(x) = M(x, x1) * h1(x) # Define the deflated function
x2 = newton(h2, x0) # Find the second root with the same intial guess
h3(x) = M(x, x2) * h2(x) # Define the second deflated function
x3 = newton(h3, x0)
sol = [x1, x2, x3]

# Define a function to test whether r = (0, 0) is a magnetic zero
function mag_zero()
    success = 0
    fail = 0
    for i = -1: 0.1: 1
        for j = -1: 0.1: 1
            x0 = [i; j]
            x1 = newton(h1, x0)
            if norm(x1 - [0; 0]) < 1e-10
                success += 1
            else
                fail += 1
            end
        end
    end
    println(success)
    println(fail)
end

mag_zero()



# The 3x3 system
function f(x)
    return sin(x[1]) + cos(x[2])
end

n = 100
x = range(-1, 1, n)
y = range(-1,1,n)

xgrid = repmat(x',n,1)
ygrid = repmat(y,1,n)

z = zeros(n,n)

for i in 1:n
    for j in 1:n
        z[i:i,j:j] = f([x[i],y[j]])
    end
end

plot_wireframe(xgrid,ygrid,z)

## new line




# Figure 1.3
f1(x) = sin(x)
p = plot(f1, -1, 8, label=L"y=\sin(x)")
f2(x) = sin(x) / x
plot!(f2, -1, 8, label=L"y=\frac{\sin(x)}{x}")
f2(x) = sin(x) / (x * (x-π))
plot!(f2, -1, 8, label=L"y=\frac{\sin(x)}{x(x-\pi)}")
hline!([0], label="", color="black")