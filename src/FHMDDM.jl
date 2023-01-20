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
export  Characterization,
        concatenateparameters,
        crossvalidate,
        drawsamples,
        expectedemissions,
        indexparameters,
        initializeparameters!,
        initialize_for_stochastic_transition!,
        Model,
        maximize_choice_posterior!,
        maximizeposterior!,
        maximizeevidence!,
        ModelSummary,
        ∇∇loglikelihood,
        learnparameters!,
        posterior_first_state,
        simulate,
        save,
        savedata,
        test
include("types.jl") # This list contains files that in which functions and types are specified. The file "types.jl" has to be listed first, but the order of the other files does not matter.
include("accumulatortransformation.jl")
include("characterization.jl")
include("choicemodel.jl")
include("conversions.jl")
include("crossvalidation.jl")
include("drift_diffusion_dynamics.jl")
include("E_step.jl")
include("evidenceoptimization.jl")
include("gaussianprior.jl")
include("Hessian.jl")
include("loadmodel.jl")
include("maximumlikelihood.jl")
include("mixture_of_Poisson_GLM/accumulatortransformation.jl")
include("mixture_of_Poisson_GLM/mixture_of_Poisson_GLM.jl")
include("mixture_of_Poisson_GLM/parameterinitialization.jl")
include("mixture_of_Poisson_GLM/parametersorting.jl")
include("mixture_of_Poisson_GLM/Poisson.jl")
include("mixture_of_Poisson_GLM/temporal_basis_functions.jl")
include("parameterlearning.jl")
include("parametersorting.jl")
include("sampling.jl")
include("save.jl")
include("tests.jl")
end
