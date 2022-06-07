# Goal 9 Matrix

using LinearAlgebra
import LinearAlgebra: factorize


function FD2D(n)
    A = zeros((n-1)^2,(n-1)^2)
    for k = 1:(n-1)^2
        A[k,k] = 4
    end
    for m = 1:(n-1)^2-1
        A[m+1, m] = -1
        A[m, m+1] = -1
    end
    for p = 1:(n-1)^2-3
        A[p+3, p] = -1
        A[p, p+3] = -1
    end
    A
end


B = FD2D(4)
B = FD2D(5) # Incorrect

# function that computes the finite difference in 2D
function fd_2d(n)
    n -= 1
    A = Tridiagonal(fill(-1, n-1), fill(2, n), fill(-1, n-1))
    kron(A, I(n)) + kron(I(n), A)
end

fd_2d(4)
fd_2d(9)