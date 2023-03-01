# For all the adaptive methods we use the step size parameter only to compute an initial forward step and then use this new point to estimate the local lipschitz constant
using LinearAlgebra: norm

include("utils.jl")


function eg(VI::AbstractVI, params::ProblemParams, cb::Callback; γ::Float64=1., adaptive=false, universal=false, ada_heuristic=false)
    """Extra-gradient method for VI given by F with option for adaptive step size."""

    F, Π = VI.F, VI.Π
    z = z0 = params.z0
    lr = 1/params.L

    if adaptive
        # more practical way to get a good initial learning rate
        u = Π(z -  lr * F(z))
        lr = norm(z-u)/ norm(F(z)-F(u))
    end
    sumofsquares = 0
    cb(VI, z, lr, F(z), γ, 0.)

    k = 0
    while k <= params.n_grad_eval
        # extrapolation step
        Fz = F(z)
        u = Π(z - lr*Fz)

        # update step
        Fu = F(u)
        z_new = Π(z - γ*lr*Fu)

        # update stepsize
        if adaptive
            lr = min(lr, norm(u-z)/norm(Fu-Fz))
            if ada_heuristic
                lr = min(lr, norm(u-z_new)/norm(Fu-F(z_new)))
            end
        end
        sumofsquares += norm(Fz-Fu)^2
        if universal; lr = 1/sqrt(1+sumofsquares) end

        z = z_new
        k+=2
        cb(VI, u, lr, Fu, γ, k)
    end
end


function ogda(VI::AbstractVI, params::ProblemParams, cb::Callback; γ::Float64=1., adaptive=false)
    """Optimistic GDA for VI given by F with option for adaptive step size"""

    F, Π = VI.F, VI.Π
    z = z0 = params.z0

    if γ == 1. # regular OGDA
        lr = 1/(2*params.L)
    else # OGDA+
        lr = (1-γ)/((1+γ)*params.L)
    end

    F_old = F(z)

    if adaptive
        z_new = Π(z - lr*F(z))
        lr = 1/2 * norm(z_new - z)/ norm(F(z_new)-F(z))
    else
        F_new = F_old
    end
    cb(VI, z0, lr, F_old, γ, 0)

    k = 0
    while k <= params.n_grad_eval
        # TODO: with nonconstant stepsize this should be modified
        z_new = Π(z - lr*((1+γ)*F_new - F_old))

        if adaptive; lr = min(lr, 1/2 * norm(z_new - z)/norm(F_new - F_old)) end
        F_old, z = F_new, z_new
        F_new = F(z)

        k+=1
        cb(VI, z_new, lr, F_new, γ, k)
    end
end


function forward_backward(VI::AbstractVI, params::ProblemParams, cb::Callback)

    F, Π = VI.F, VI.Π
    z = params.z0
    lr = 0.1/params.L

    cb(VI, z, lr, F(z), 1, 0.)

    k = 0
    while k <= params.n_grad_eval
        Fz = F(z)
        z_new = Π(z - lr*Fz)

        z = z_new
        k+=1
        cb(VI, z, lr, Fz, 1, k)
    end
end


function adaptive_EG(VI::AbstractVI, params::ProblemParams, cb::Callback; ν=0.99, τ=0.9, backtrack=true, count_grad_eval=true, α=missing)
    """Extra-gradient method from the paper 'escaping' limit cycles. Designed
    for weak Minty problems.  Adaptively computes a reduction in the update step
    compared to the extrapolation step.
    """

    ρ = -params.ρ/2 # use their definition of ρ
    if ismissing(ρ); ρ = -Inf end
    F, Π = VI.F, VI.Π
    z, u = params.z0, params.z0
    lr = 1/params.L # generous starting value so it does not start too small
    Fu = F(u)

    # some dummy values so the first callback is ok
    α = 1/2

    k = 0
    while k <= params.n_grad_eval

        # update stepsize
        if backtrack; lr, eval_bt = backtracking_ls(F, Π, z, lr, ν, τ, true) end
        cb(VI, u, lr, Fu, α, k)

        # update iterates
        Fz = F(z)
        u = Π(z - lr*Fz)
        Fu = F(u)
        Hu = u - lr*Fu
        Hz = z - lr*Fz

        # if theoretical bound (lr > -2ρ) not satisfied use smallest acceptable
        tmp = max(-0.499*lr, ρ)
        α = tmp/lr + dot(u-z, Hu-Hz) / norm(Hu-Hz)^2

        z = z + α*(Hu - Hz)

        k+= 2
        if count_grad_eval k+= eval_bt end

    end
    cb(VI, u, lr, Fu, α, k)
end

function curvature_ls(F, z, ν=0.99)
    J(u) = ForwardDiff.jacobian(F, u)
    norm_hessian(u) = opnorm(J(u))
    if any(isnan.(z))
        return 0
        # TODO: should actually stop computation
    else
        lr_init = ν/norm_hessian(z)
    end

    return lr_init
end


function backtracking_ls(F, Π, z, lr::Float64, ν=0.9, τ=0.9, curvature=true)

    k = 0  # count extra gradient evals due to backtracking
    if curvature
        lr = curvature_ls(F, z)
    else # try to increase first

        increase = false
        Fz = F(z)
        Gz = Π(z-lr*Fz)
        # if we start with feasible stepsize, increase to maximum
        while lr * norm(F(Gz) -Fz) <= (ν * norm(Gz - z)) && k < 20
            increase = true
            lr = lr * 1/τ
            Gz = Π(z-lr*Fz)
            k += 1
        end
        # if we increased, stop. no need to do bt in the other direction
        if increase; return lr*τ, k end
    end

    Fz = F(z)
    Gz = Π(z-lr*Fz)
    # if stepsize is not feasible, decrease until it is
    while true
        if lr * norm(F(Gz) -Fz) <= (ν * norm(Gz - z)) || k > 20
            return lr, k
        end
        lr = τ * lr
        Gz = Π(z-lr*Fz)
        k += 1
    end
end
