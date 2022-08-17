module FHMDDM
using   Distributed,
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
import  Flux
export  concatenateparameters,
        crossvalidate,
        expectedemissions,
        initializeparameters!,
        initialize_for_stochastic_transition!,
        Model,
        maximize_choice_posterior!,
        maximizeposterior!,
        maximizeevidence!,
        ∇∇loglikelihood,
        learnparameters!,
        posterior_first_state,
        save,
        test
include("types.jl") # This list contains files that in which functions and types are specified. The file "types.jl" has to be listed first, but the order of the other files does not matter.
include("accumulatortransformation.jl")
include("choicemodel.jl")
include("conversions.jl")
include("crossvalidation.jl")
include("drift_diffusion_dynamics.jl")
include("E_step.jl")
include("evidenceoptimization.jl")
include("gaussianprior.jl")
include("loadmodel.jl")
include("maximumlikelihood.jl")
include("mixture_of_Poisson_GLM/mixture_of_Poisson_GLM.jl")
include("mixture_of_Poisson_GLM/parameterinitialization.jl")
include("mixture_of_Poisson_GLM/accumulatortransformation.jl")
include("parameterlearning.jl")
include("parametersorting.jl")
include("temporal_basis_functions.jl")
include("sampling.jl")
include("save.jl")
include("tests.jl")
include("two_pass_Hessian.jl")
end
