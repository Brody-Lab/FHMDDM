module FHMDDM
using   Bootstrap,
        Distributed,
        Distributions,
        DSP,
        ForwardDiff,
        LinearAlgebra,
        LineSearches,
        MAT,
        Optim,
        Parameters,
        Random,
        SpecialFunctions,
        StatsFuns
import  CSV,
        DataFrames
export  analyzeandsave,
        Characterization,
        concatenateparameters,
        crossvalidate,
        expectedemissions,
        indexparameters,
        initializeparameters!,
        Model,
        maximize_choice_posterior!,
        maximizeposterior!,
        maximizeevidence!,
        ModelSummary,
        ∇∇loglikelihood,
        Options,
        learnparameters!,
        simulate,
        simulateandsave,
        sortparameters!,
        save,
        test
include("types.jl") # This list contains files that in which functions and types are specified. The file "types.jl" has to be listed first, but the order of the other files does not matter.
include("accumulatortransformation.jl")
include("characterization.jl")
include("choicemodel.jl")
include("crossvalidation.jl")
include("drift_diffusion_dynamics.jl")
include("drift_diffusion_parameters.jl")
include("E_step.jl")
include("evidenceoptimization.jl")
include("gaussianprior.jl")
include("hessian.jl")
include("loadmodel.jl")
include("maximumlikelihood.jl")
include("mixture_of_Poisson_GLM/accumulatortransformation.jl")
include("mixture_of_Poisson_GLM/basicglm.jl")
include("mixture_of_Poisson_GLM/driftbasis.jl")
include("mixture_of_Poisson_GLM/hessian.jl")
include("mixture_of_Poisson_GLM/inverselink.jl")
include("mixture_of_Poisson_GLM/loadmodel.jl")
include("mixture_of_Poisson_GLM/mixture_of_Poisson_GLM.jl")
include("mixture_of_Poisson_GLM/parameterinitialization.jl")
include("mixture_of_Poisson_GLM/parametersorting.jl")
include("mixture_of_Poisson_GLM/Poisson.jl")
include("mixture_of_Poisson_GLM/temporal_basis_functions.jl")
include("mixture_of_Poisson_GLM/decoupling.jl")
include("parameterlearning.jl")
include("parametersorting.jl")
include("perievent_time_histogram.jl")
include("realnative.jl")
include("sampling.jl")
include("save.jl")
include("tests.jl")
end
