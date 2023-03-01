using LinearAlgebra
include("../utils/utils.jl")

function forsaken_difficult(n_grad_eval=400)

    L_inverse = 0.08
    L = 1/L_inverse
    ρ = 0.477761
    z0 = [0.5, 0.5]
    sol = [0.0780267, 0.411934]

    folder_name = "Forsaken"

    ψ(x) = x/2 - 2x^3 + x^5
    ψ_p(x) = 1/2 - 6x^2 + 5x^4

    function F(u)
        x, y = u[1], u[2]
        return [ψ(x) + (y-0.45); -x + ψ(y)]
    end

    proj(u) = clamp.(u, -10, 10)

    problem = VariationalInequality(F, proj, sol=sol)
    params = ProblemParams(z0, n_grad_eval, folder_name, L, ρ=ρ)

    return problem, params
end
