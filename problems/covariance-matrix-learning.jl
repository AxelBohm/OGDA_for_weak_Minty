using LinearAlgebra
using Random
using Distributions

include("../utils/utils.jl")

function covariance_matrix_learning()
    folder_name = "covariance"
    Random.seed!(42)

    # dimension
    d = 10
    # number of summands
    samples = 128

    # lr = 0.02
    L = 50 # guess
    n_grad_eval = 1000

    proj(u) = u
    F, z0 = generate_cov_F(d, samples)
    covariance = VariationalInequality(F, proj, sol=missing, dim=2d^2)
    params = ProblemParams(z0, n_grad_eval, folder_name, L)

    return covariance, params
end

# need to wrap stuff into functions
function generate_cov_F(d, samples)

    # random covariance matrix
    A = rand(Normal(), d, d) +  5*I
    Σ = A * A'
    # Σ = Matrix{Float64}(I, d, d)

    # generator input
    # number of input samples does not need to match the ones of the observed samples
    zz = rand(Normal(0, 1), d, samples)

    # observed samples
    # xx = rand(MvNormal(zeros(d), Σ), samples)
    xx = A * zz

    function nabla_x(θ, ϕ)
        tmp2 = zeros(Float64, d, d)
        for i in 1:samples
            tmp2 += zz[:,i] * zz[:,i]'
        end
        return - (ϕ + ϕ') * θ * tmp2 /samples
    end

    function nabla_y(θ, ϕ)
        tmp = zeros(Float64, d, d)
        tmp1 = zeros(Float64, d, d)
        for i in 1:samples
            tmp1 += xx[:, i] * xx[:, i]'
            tmp += zz[:, i] * zz[:, i]'
        end

        return (tmp1 -θ * tmp * θ')/samples
    end

    function F(u)
        return [nabla_x(u[1:d, :], u[d+1:end, :]); - nabla_y(u[1:d, :], u[d+1:end, :])]
    end

    skew = ones(d,d) - 2UpperTriangular(ones(d,d))+I
    z0 = [A+ 0.3*rand(d,d); skew + 0.3*rand(d,d)]
    return F, z0
end