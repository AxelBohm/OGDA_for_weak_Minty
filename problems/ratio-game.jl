using LinearAlgebra

include("../utils/utils.jl")

function ratio_game()
    ϵ = 0.49
    s = 0.5

    R = [-1 ϵ; -ϵ 0]
    S = [s s; 1 1.0]

    folder_name  = "epsilon=$ϵ s=$s"

    n_grad_eval = 300
    L = 10 # guessed
    ρ = 1/8 # where did I get this?
    z0 = [1., 0., 1., 0.]
    sol = [0., 1., 0., 1.]

    # setup problem
    F = generate_F(R, S)
    problem = VariationalInequality(F, proj_twice_simplex, sol=sol)
    params = ProblemParams(z0, n_grad_eval, folder_name, L, ρ=ρ)

    return problem, params
end