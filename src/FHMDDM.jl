module FHMDDM

using   Distributed, # External packages that we use
        Distributions,
        ForwardDiff,
        LinearAlgebra,
        LineSearches,
        MAT,
        Optim,
        Parameters,
        Random,
        ReverseDiff,
        SpecialFunctions,
        StatsFuns

export  adapt, # module-specific functions and types in this module that we can call in the REPL without preceding with the name of the module
        approximatetransition!,
        Clicks,
        concatenatebounds,
        concatenateparameters,
        concatenate_choice_related_parameters,
        conditional_probability_of_choice,
        conditionedmean,
        estimatefilters,
        estimatefilters!,
        expectedemissions,
        Options,
        forward,
        𝐇negativeexpectation,
        Indexθ,
        initializeparameters,
        initializeparameters!,
        Latentθ,
        lambda,
        likelihood,
        likelihood!,
        loglikelihood,
        loglikelihood!,
        loglikelihoodchoices,
        MixturePoissonGLM,
        Model,
        maximizechoiceLL!,
        maximizelikelihood!,
        ∇adapt,
        ∇loglikelihood,
        ∇negativeloglikelihood!,
        ∇negativeexpectation,
        ∇negativeexpectation!,
        native2real,
        native2real!,
        negativeexpectation,
        posteriors,
        probabilityvector,
        probabilityvector!,
        real2native,
        real2native!,
        Shared,
        sample,
        save,
        savedata,
        sortparameters,
        sortparameters!,
        stochasticmatrix!,
        temporal_bases_values,
        Trial,
        Trialinvariant,
        Trialset,
        update!
include("types.jl") # This list contains files that in which functions and types are specified. The file "types.jl" has to be listed first, but the order of the other files does not matter.
include("automaticdifferentiation.jl")
include("benchmarking.jl")
include("choicemodel.jl")
include("conversions.jl")
include("drift_diffusion_dynamics.jl")
include("loadmodel.jl")
include("maximumlikelihood.jl")
include("mixture_of_Poisson_GLM.jl")
include("temporal_basis_functions.jl")
include("sampling.jl")
include("save.jl")
#include("testing.jl")

end # module
