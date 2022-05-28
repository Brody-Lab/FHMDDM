module FHMDDM

using   Distributed, # packages whose name and exported function is in scope
        Distributions,
        ForwardDiff,
        LinearAlgebra,
        LineSearches,
        MAT,
        Optim,
        Parameters,
        Random,
        SpecialFunctions,
        StatsFuns
import  Flux # packages whose name but not its exported function is in scope
export  crossvalidate, # module-specific functions and types in this module that we can call in the REPL without preceding with the name of the module
        expectedemissions,
        initializeparameters!,
        initialize_for_stochastic_transition!,
        Model,
        maximize_choice_posterior!,
        maximizeposterior!,
        posterior_first_state,
        save
include("types.jl") # This list contains files that in which functions and types are specified. The file "types.jl" has to be listed first, but the order of the other files does not matter.
include("choicemodel.jl")
include("conversions.jl")
include("crossvalidation.jl")
include("drift_diffusion_dynamics.jl")
include("E_step.jl")
include("evidence_optimization.jl")
include("loadmodel.jl")
include("maximumlikelihood.jl")
include("maximumaposteriori.jl")
include("mixture_of_Poisson_GLM.jl")
include("parametersorting.jl")
include("temporal_basis_functions.jl")
include("sampling.jl")
include("save.jl")
include("two_pass_Hessian.jl")

end # module
