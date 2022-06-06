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