using LinearAlgebra
using Test
using Base

# Captures problem instances
abstract type AbstractVI end

Base.@kwdef struct VariationalInequality <: AbstractVI
    dim::Int64
    F::Function
    Π::Function
    sol = missing
    xdim::Int64

    # constructor
    function VariationalInequality(F::Function, Π::Function; sol=missing, xdim::Union{Missing,Int64}=missing,
                                   dim::Union{Missing,Int64}=missing)
        # if not specified assume that xdim==ydim
        if !ismissing(sol)
            dim = size(sol)[1]
        elseif ismissing(dim)
            err("If no solution is given, dimension needs to be specified.")
        end
        if ismissing(xdim)
            xdim = Int(dim/2)
        end

        return new(dim, F, Π, sol, xdim)
    end
end


function mk_folder_structure(folder_name)
    path = "../output/$folder_name"
    mkpath(path)
    return path
end


function simplex_projection(x)
    """Projection onto the unit simplex."""

    u = sort(x, rev=true)
    ρ = 1
    i = 1
    for i in 1:length(x)
        if u[i] + 1/i * (1 - sum(u[1:i])) > 0
            ρ = i
        end
    end
    λ = 1/ρ * (1 - sum(u[1:ρ]))
    out = max.(x .+ λ, 0)
    return out
end


function proj_twice_simplex(z)
    "Projection onto twice the unit simplex, i.e. Δ × Δ. "
    # TODO: this assumes that xdim == dim/2 which I do not assume anywhere else
    n = Int(length(z)/2)
    return [simplex_projection(z[1:n]); simplex_projection(z[n+1:end])]
end


function nabla_x(x, y, R, S)
    """Partial derivative with respect to _first_ component of the ratio game."""
    return (R * y .* (x' * S * y) - S * y .* (x' * R * y) ) ./ (x' * S * y)^2
end

function nabla_y(x, y, R, S)
    """Partial derivative with respect to _second_ component of the ratio game."""
    return (R' * x .* (x' * S * y) - S' * x .* (x' * R * y) ) ./ (x' * S * y)^2
end


function generate_F(R, S)
    """Generate F for ratio game."""

    function F(z)
        """Stacked gradients according to VI formulation."""
        return [nabla_x(z[1:2], z[3:4], R, S); -nabla_y(z[1:2], z[3:4], R, S)]
    end
    return F
end


Base.@kwdef struct Callback
    """Can be passed to an Algorithm and is supposed to be called after every iteration to store
     various optimality measures. """
    label::String
    x_ticks::Vector{Float64} = []
    iterates::Array = []
    store_grad::Vector{Float64} = []
    vi_val::Vector{Float64} = []
    lr::Vector{Float64} = []
    lr_ratio::Vector{Float64} = []
end

function (cb::Callback)(VI::AbstractVI, u, lr, Fu, lr_ratio, k)
    "Makes the Callback struct actually callable and computes various optimality measures "
    append!(cb.store_grad, compute_norm_grad(VI, u, lr, Fu))
    if !ismissing(VI.sol)
        append!(cb.vi_val, -2*eval_vi(VI, u, Fu)/norm(Fu)^2/lr)
    end
    append!(cb.x_ticks, k)
    if VI.dim <= 4
        append!(cb.iterates, u)
    end
    append!(cb.lr, lr)
    append!(cb.lr_ratio, lr_ratio)
end

function compute_norm_grad(VI::AbstractVI, u, lr, Fu=missing)
    """Compute fixed point residual."""
    if ismissing(Fu); Fu = VI.F(u) end
    Π = VI.Π
    return norm(Π(u - lr*Fu) - u)^2/lr^2
end

function eval_vi(VI::AbstractVI, u, Fu=missing)
    """Computes a quantity that should be positive for monotone problems."""
    if ismissing(Fu); Fu = VI.F(u) end
    sol = VI.sol
    return Fu' * (u - sol)
end

Base.@kwdef mutable struct ProblemParams
    z0::Array{Float64}
    n_grad_eval::Int64
    path::String
    L = Inf
    # (estimate of) weak Minty parameter
    ρ = missing

    function ProblemParams(z0::Array{Float64}, n_grad_eval::Int64, folder_name::String, L=Inf; ρ=missing)
        path = mk_folder_structure(folder_name)
        return new(z0, n_grad_eval, path, L, ρ)
    end
end

Base.@kwdef struct AlgParams
    γ::Float64
    α::Float64
    lr::Float64
    ϕ::Float64
    callback::Callback
end

mutable struct algorithm
    method::Function
    label::String
    params::NamedTuple
    # constructor
    function algorithm(method::Function, label::String, params=(;))
        return new(method, label, params)
    end
end
