import Base.broadcast: broadcasted
n = 100

struct JacobianDual{T}
    a::T
    B::T
end

JacobianDual(a::AbstractVector) = JacobianDual(a, zeros(length(a), length(a)))

const ϵ = JacobianDual(zeros(n), zeros(n, n))

*(Δ::AbstractVector, )