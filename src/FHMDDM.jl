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
export  adapt, # module-specific functions and types in this module that we can call in the REPL without preceding with the name of the module
        approximatetransition!,
        Clicks,
        choicelikelihood,
        choiceposteriors,
        concatenatebounds,
        concatenateparameters,
        concatenate_choice_related_parameters,
        conditional_probability_of_choice,
        conditionedmean,
        differentiateℓ_wrt_ψ,
        drectifya,
        dtransformaccumulator,
        estimatefilters,
        estimatefilters!,
        expectedemissions,
        Options,
        forward,
        GLMθ,
        𝐇negativeexpectation!,
        Indexθ,
        initializeparameters,
        initializeparameters!,
        Latentθ,
        likelihood,
        likelihood!,
        linearpredictor,
        loglikelihood,
        loglikelihood!,
        loglikelihoodchoices,
        MixturePoissonGLM,
        Model,
        maximizechoiceLL!,
        maximizelikelihood!,
        maximizeposterior!,
        ∇adapt,
        ∇loglikelihood,
        ∇negativeloglikelihood,
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
        rectifya,
        Shared,
        sample,
        sampleemissions,
        save,
        sortparameters,
        sortparameters!,
        stochasticmatrix!,
        temporal_bases_values,
        transformaccumulator,
        Trial,
        Trialinvariant,
        Trialset,
        update!
include("types.jl") # This list contains files that in which functions and types are specified. The file "types.jl" has to be listed first, but the order of the other files does not matter.
include("benchmarking.jl")
include("choicemodel.jl")
include("conversions.jl")
include("drift_diffusion_dynamics.jl")
include("E_step.jl")
include("loadmodel.jl")
include("maximumlikelihood.jl")
include("maximumaposteriori.jl")
include("mixture_of_Poisson_GLM.jl")
include("parametersorting.jl")
include("temporal_basis_functions.jl")
include("sampling.jl")
include("save.jl")

end # module
