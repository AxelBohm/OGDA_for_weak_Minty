using LinearAlgebra

include("../utils/utils.jl")

function global_forsaken()
    """Is weak minty with \rho > -1/L."""

    n_grad_eval = 300
    L_inverse = 0.360726
    L = 1/L_inverse
    z0 = [1., 1.]
    sol = [0, 0]
    ρ = 0.119732

    folder_name = "GlobalForsaken"

    ψ(x) = 4x^5/7 - 4x^3/3 + 2x/3
    ψ_p(x) = 20x^4/7 - 4x^2 + 2/3
    function F(u)
        x, y = u[1], u[2]
        return [ψ(x) + y; -x + ψ(y)]
    end
    proj(u) = clamp.(u, -1.5, 1.5)
    problem = VariationalInequality(F, proj, sol=sol)
    params = ProblemParams(z0, n_grad_eval, folder_name, L, ρ=ρ)

    return problem, params
end
