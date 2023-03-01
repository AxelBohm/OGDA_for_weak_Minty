include("../utils/utils.jl")
include("../utils/runner.jl")
include("../utils/methods.jl")

include("../problems/problems.jl")

# this is the comparison used for the paper

# shows that OGDA+ can beat EG+
algorithms = [algorithm(eg, "EG+ γ=0.01", (; γ=0.01)),
              algorithm(ogda, "OGDA+ γ=0.1", (; γ=0.1))]
run_instance(algorithms, () -> lower_bound(3))


# shows competitive performance of EG+ adaptive
algorithms =[algorithm(eg, "EG+", (; γ=1/2)),
             algorithm(ogda, "OGDA+", (; γ=1/2)),
             algorithm(eg, "EG+ adaptive", (; γ=1/2, adaptive=true)),
             algorithm(adaptive_EG, "CurvatureEG+")]

run_instance(algorithms, ratio_game_exotic)
run_instance(algorithms, forsaken_difficult)

# shows the need to reduce γ
algorithms = [algorithm(ogda, "OGDA+ γ=0.3", (; γ=0.3)),
              algorithm(ogda, "OGDA", (; γ=1.))]
run_instance(algorithms, () -> polar_game(1/3))
