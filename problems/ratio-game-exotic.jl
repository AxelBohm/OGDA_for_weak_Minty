using LinearAlgebra

include("../utils/utils.jl")

function ratio_game_exotic()
    R = [-0.6 -0.3; 0.6 -0.3]
    S = [0.9 0.5; 0.8 0.4]

    n_grad_eval = 500
    L = 5/3 # guessed
    z0 = [1., 0., 1., 0.]
    # z0 = [0.2, 0.8, 0.8, 0.2]
    xs = 0.951941
    ys = 0.050485
    sol = [xs, 1-xs, ys, 1-ys]

    folder_name = "ratio-game-exotic"

    F = generate_F(R, S)
    problem = VariationalInequality(F, proj_twice_simplex, sol=sol)
    params = ProblemParams(z0, n_grad_eval, folder_name, L)

    return problem, params
end