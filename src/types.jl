"""
	Latentθ

Parameters of the latent variables in the factorial hidden Markov drift-diffusion model.

Not included are the weights of the linear filters of the mixture of Poisson generalized linear model of each neuron
"""
@with_kw struct Latentθ{VR}
	"transition probability of the coupling variable to remain in the coupled state"
	Aᶜ₁₁::VR=[NaN]
	"transition probability of the coupling variable to remain in the uncoupled state"
	Aᶜ₂₂::VR=[NaN]
	"height of the sticky bounds"
	B::VR=[NaN]
	"exponential change rate of inter-click adaptation"
	k::VR=[NaN]
	"leak or instability"
	λ::VR=[NaN]
	"constant added to the mean of the distribution of the accumulator variable at the first time step"
	μ₀::VR=[NaN]
	"strength of inter-click adaptation and sign of the adaptation (facilitation vs. depression)"
	ϕ::VR=[NaN]
	"prior probability of the coupling variable in the coupled state"
	πᶜ₁::VR=[NaN]
	"prior probability that the accumulator variable is not used to determine the behavioral choice"
	ψ::VR=[NaN]
	"multiplied by the width of the timestep `Δt`, this is the variance of the Gaussian noise added at each time step"
	σ²ₐ::VR=[NaN]
	"variance of the prior probability of the accumulator variable"
	σ²ᵢ::VR=[NaN]
	"multiplied by the sum of the absolute value of the post-adaptation click input, this is the variance of Gaussian noise added as a result of the clicks"
	σ²ₛ::VR=[NaN]
	"weight of the rewarded option of the previous trial on the mean of the accumulator at the first time step"
	wₕ::VR=[NaN]
end

"""
    FHMDDMoptions

Model settings

"""
@with_kw struct FHMDDMoptions{TB<:Bool,
							  TS<:String,
							  TF<:AbstractFloat,
							  TVF<:Vector{<:AbstractFloat},
							  TVI<:Vector{<:Integer},
							  TI<:Integer}
	"number of temporal basis functions for the accumulator per s"
    a_basis_per_s::TI=10
	"response latency of the accumulator to the clicks"
    a_latency_s::TF=1e-2
	"type of temporal basis functions"
    basistype::TS="raised_cosine"
	"full path of the data"
    datapath::TS=""
	"duration of each timestep in seconds"
    Δt::TF=1e-2
	"number of states of the coupling variable"
    K::TI = 2; 			@assert K == 1 || K == 2
	"whether to fit the height of the sticky bounds"
	fit_B::TB=true
	"whether to fit the exponential change rate of inter-click adaptation"
	fit_k::TB=true
	"whether to fit the parameter specifying leak or instability"
	fit_λ::TB=true
	"whether to fit the constant added to the mean of the distribution of the accumulator variable at the first time step"
	fit_μ₀::TB=true
	"whether to fit the strength of inter-click adaptation and sign of the adaptation (facilitation vs. depression)"
	fit_ϕ::TB=true
	"whether to fit the behavioral lapse rate"
	fit_ψ::TB=true
	"whether to fit the variance of the Gaussian noise added at each time step"
	fit_σ²ₐ::TB=true
	"whether to fit the variance of the prior probability of the accumulator variable"
	fit_σ²ᵢ::TB=true
	"whether to fit the variance of Gaussian noise added as a result of the clicks"
	fit_σ²ₛ::TB=true
	"whether to fit the weight of the rewarded option of the previous trial on the mean of the accumulator at the first time step"
	fit_wₕ::TB=true
	"value in native space of the transition probability of the coupling variable to remain in the coupled state that corresponds to zero in real space"
	q_Aᶜ₁₁::TF=1-1e-3; 	@assert q_Aᶜ₁₁ >= 0 && q_Aᶜ₁₁ <= 1
	"value in native space of the transition probability of the coupling variable to remain in the uncoupled state that corresponds to zero in real space"
	q_Aᶜ₂₂::TF=1-1e-3; 	@assert q_Aᶜ₂₂ >= 0 && q_Aᶜ₂₂ <= 1
	"value of the bound height in native space that corresponds to zero in real space"
	q_B::TF=30.0; 		@assert q_B > 0
	"value of the adaptation change rate in native space that corresponds to zero in real space"
	q_k::TF=1e-3; 		@assert q_k > 0
	"value in native space of the prior probability of the coupling variable in coupled state that corresponds to zero in real space"
	q_πᶜ₁::TF=1-1e-3; 	@assert q_πᶜ₁ >= 0 && q_πᶜ₁ <= 1
	"value in native space of the behavioral lapse rate that corresponds to zero in real space"
	q_ψ::TF=1e-3; 		@assert q_ψ >= 0 && q_ψ <= 1
	"value in native space of the variance of per-timestep noise that corresponds to zero in real space"
	q_σ²ₐ::TF=1e-3; 	@assert q_σ²ₐ >= 0
	"value in native space of the variance of the initial probability of the accumulator variable that corresponds to zero in real space"
	q_σ²ᵢ::TF=1e-3; 	@assert q_σ²ᵢ >= 0
	"value in native space of the variance of the variance of per-click noise that corresponds to zero in real space"
	q_σ²ₛ::TF=1e-3;	 	@assert q_σ²ₛ >= 0
	"lower and upper bounds of the probability of remaining in the coupled state"
	bounds_Aᶜ₁₁::TVF=[logit(1e-6)-logit(q_Aᶜ₁₁); logit(1-1e-6)-logit(q_Aᶜ₁₁)]
	"lower and upper bounds of the probability of remaining in the coupled state"
	bounds_Aᶜ₂₂::TVF=[logit(1e-6)-logit(q_Aᶜ₂₂); logit(1-1e-6)-logit(q_Aᶜ₂₂)]
	"lower and upper bounds of the sticky bound height in real space"
	bounds_B::TVF=[-Inf; Inf]
	"lower and upper bounds of the adaptation exponential change rate in real space"
	bounds_k::TVF=[-Inf; Inf]
	"lower and upper bounds of the leakiness/instability parameter"
	bounds_λ::TVF=[-Inf; Inf]
	"lower and upper bounds of the mean of the prior probability"
	bounds_μ₀::TVF=[-Inf; Inf]
	"lower and upper bounds of the adaptation strength"
	bounds_ϕ::TVF=[-Inf; Inf]
	"lower and upper bounds of the initial probability of the coupled state in real space"
	bounds_πᶜ₁::TVF=[logit(1e-6)-logit(q_πᶜ₁); logit(1-1e-6)-logit(q_πᶜ₁)]
	"lower and upper bounds of the lapse rate in real space"
	bounds_ψ::TVF=[logit(1e-6)-logit(q_ψ); logit(1-1e-6)-logit(q_ψ)]
	"lower and upper bounds of the diffusion noise"
	bounds_σ²ₐ::TVF=[-Inf; Inf]
	"lower and upper bounds of the variance of the prior probability"
	bounds_σ²ᵢ::TVF=[-Inf; Inf]
	"lower and upper bounds of the variance of the per-click noise"
	bounds_σ²ₛ::TVF=[-Inf; Inf]
	"lower and upper bounds of the weight of the previous rewarded option"
	bounds_wₕ::TVF=[-Inf; Inf]
	"where the results of the model fitting are to be saved"
    resultspath::TS=""
	"the number of time bins before the current bin when the spike history is considered, one value for each regressor, such as [1, 2, ..., 9]. Note a positive lag represents a time bin before the current time bin."
    spikehistorylags::TVI=collect(1:10)
    "number of states of the discrete accumulator variable"
    Ξ::TI=53; @assert isodd(Ξ) && Ξ > 1
end

"""
    Clicks

Information on the clicks delivered during one trial

The stereoclick is excluded.
"""
@with_kw struct Clicks{VF<:Vector{<:AbstractFloat},
                       BA1<:BitArray{1},
					   VI<:Vector{<:Integer},
                       VVI<:Vector{<:Vector{<:Integer}}}
    "A vector of floats indicating the time of each click. It is sorted in ascending order."
    time::VF
	"A vector of integers indicating the timesteps with clicks. Not the timestep of each click"
	inputtimesteps::VI
	"A vector of integers whose element `i = inputindex[t]` indicating the i-th input for that time step. `inputtimesteps[inputindex[t]] == t` and `inputindex[inputtimesteps[i]] == i`"
	inputindex::VVI
    "An one dimensional BitArray specifying the side of each click. The times of the right and left clicks are given by calling `time[source]` and `time[.!source]`, respectively. "
    source::BA1
    "A vector of integers indexing the left clicks that occured in each timestep. The times of the left clicks in the t-th timestep can be found by calling `time[left[t]]`."
    left::VVI
    "A vector of integers indexing the right clicks that occured in each timestep. The times of the right click in the t-th timestep can be found by calling `time[right[t]]`."
    right::VVI
end

"""
    Trial

Information on the sensory stimulus and behavior each trial

Spike trains are not included. In sampled data, the generatives values of the latent variables are stored.
"""
@with_kw struct Trial{TB<:Bool,
                      TC<:Clicks,
                      TI<:Integer,
                      VI<:Vector{<:Integer}}
    "information on the auditory clicks"
    clicks::TC
    "behavioral choice"
    choice::TB
    "number of time steps in this trial. The duration of each trial is from the onset of the stereoclick to the end of the fixation period"
    ntimesteps::TI
    "location of the reward baited in the previous trial (left:-1, right:1, no previous trial:0)"
    previousanswer::TI
    "generative values of the accumulator index variable (𝐚). This is nonempty only in sampled data"
    a::VI=Vector{Int}(undef,0)
    "generative values of the coupling variable (𝐜). This is nonempty only in sampled data"
    c::VI=Vector{Int}(undef,0)
end

"""
    MixturePoissonGLM

Mixture of Poisson generalized linear model
"""
@with_kw struct MixturePoissonGLM{TF<:AbstractFloat,
                                  TI<:Integer,
                                  TVF<:Vector{<:AbstractFloat},
								  TVR,
                                  TMF<:Matrix{<:AbstractFloat}}
    "size of the time bin"
    Δt::TF
    "Number of coupling states"
    K::TI
    "Columns of the design matrix that are invariant to the accumulator variable and correspond to regressors related to spike history or timing of events. Each column is scaled such that the maximum of the absolute value is 1."
    𝐔::TMF
    "Temporal bases values. Element 𝚽[t,i] corresponds to the value of the i-th temporal basis at the t-th time bin"
    𝚽::TMF
    "Temporal bases"
    Φ::TMF
	"regression coefficients of the accumulator-independent regressors"
    𝐮::TVR=10.0 .- 20.0.*rand(size(𝐔,2))
    "Time-varying weight of the left-favoring evidence. Element 𝐥[i] corresponds to the weight of the i-th temporal basis"
    𝐥::TVR=10.0 .- 20.0.*rand(size(𝚽,2))
    "Time-varying weight of the right-favoring evidence. Element 𝐫[i] corresponds to the weight of the i-th temporal basis"
    𝐫::TVR=10.0 .- 20.0.*rand(size(𝚽,2))
	"full design matrix"
	𝐗::TMF
	"Normalized values of the accumulator"
    𝛏::TVF
    "response variable"
    𝐲::TVF
    "factorial of the response variable"
    𝐲!::TVF = factorial.(𝐲)
    "log of the factorial of `y`"
    log_𝐲!::TVF = loggamma.(𝐲.+1)
end

"""
    Trialset

A group of trials in which a population of neurons were recorded simultaneously
"""
@with_kw struct Trialset{VM<:Vector{<:MixturePoissonGLM},
						 TI<:Integer,
                         VT<:Vector{<:Trial}}
	"Mixture of Poisson GLM of each neuron in this trialset"
    mpGLMs::VM=MixturePoissonGLM[]
	"number of time steps summed across trials"
	ntimesteps::TI=size(mpGLMs[1].𝐔,1)
	"Information on the stimulus and behavior for each trial in this trial-set"
    trials::VT
	"Number of trials"
	ntrials::TI=length(trials)
end

"""
	FHMDDM

A factorial hidden Markov drift-diffusion model
"""
@with_kw struct FHMDDM{Toptions<:FHMDDMoptions,
					   Tθ1<:Latentθ,
					   Tθ2<:Latentθ,
					   Tθ3<:Latentθ,
					   VT<:Vector{<:Trialset}}
	"settings of the model"
	options::Toptions
	"model parameters in their native space (the term 'space' is not meant to be mathematically rigorous. Except for the sticky bound `B`, the native space of all parameters are positive real numbers, which is a vector space. The native space of `B` is upper bounded because I am concerned a large value of `B` would result in a loss of precision in the discretization of the accumulator variable.)"
	θnative::Tθ1
	"model parameters in real vector space ℝ"
	θreal::Tθ2
	"initial values of the parameters in native space"
	θ₀native::Tθ3
	"data used to constrain the model"
	trialsets::VT
end

"""
    Indexθ

Index of each model parameter if all values that were being fitted were concatenated into a vector
"""
@with_kw struct Indexθ{L<:Latentθ,
                       VVVI<:Vector{<:Vector{<:Vector{<:Integer}}}}
	"parameters specifying the latent variables"
	latentθ::L
    "time-varying weights of state-independent inputs to each neuron's GLM, including event timing and spike history terms"
    𝐮::VVVI=[[zeros(Int64,0)]]
    "time-varying weights of accumulated evidence in left-favoring states of the accumulator and in the coupled state"
    𝐥::VVVI=[[zeros(Int64,0)]]
    "time-varying weights of accumulated evidence in right-favoring states of the accumulator and in the coupled state"
    𝐫::VVVI=[[zeros(Int64,0)]]
end

"""
	Trialinvariant

A collection of hyperparameters and temporary quantities that are fixed across trials

"""
@with_kw struct Trialinvariant{	TI<:Integer,
								VF<:Vector{<:AbstractFloat},
								VR,
								MF<:Matrix{<:AbstractFloat},
								MR1,
								MR2,
								MR3,
							   	F<:AbstractFloat}
	"transition matrix of the accumulator variable in the absence of input"
	Aᵃsilent::MR1
	"transition matrix of the coupling variable"
	Aᶜ::MF=zeros(1,1)
	"transpose of the transition matrix of the coupling variable"
	Aᶜᵀ::MR2
	"derivitive with respect to the means of the transition matrix of the accumulator variable in the absence of input"
	dAᵃsilentdμ::MF=zeros(1,1)
	"derivitive with respect to the variance of the transition matrix of the accumulator variable in the absence of input"
	dAᵃsilentdσ²::MF=zeros(1,1)
	"derivitive with respect to the bound of the transition matrix of the accumulator variable in the absence of input"
	dAᵃsilentdB::MF=zeros(1,1)
	"time step, in seconds"
	Δt::F
	"an intermediate term used for computing the derivative with respect to the bound for the first time bin"
	𝛚::VF=zeros(1)
	"an intermediate term used for computing the derivative with respect to the bound for subsequent time bins"
	Ω::MF=zeros(1,1)
	"prior probability of the coupling variable "
	πᶜᵀ::MR3
	"discrete values of the accumulation variable"
	𝛏::VR
	"Number of coupling states"
	K::TI
	"Number of states of the discrete accumulator variable"
	Ξ::TI
end

"""
	Shared

Container of variables used by both the log-likelihood and gradient computation
"""
@with_kw struct Shared{	VF<:Vector{<:AbstractFloat},
						VVVMF<:Vector{<:Vector{<:Vector{<:Matrix{<:AbstractFloat}}}},
						TI<:Indexθ}
	"a vector of the concatenated values of the parameters being fitted"
	concatenatedθ::VF
	"a structure indicating the index of each model parameter in the vector of concatenated values"
	indexθ::TI
	"Conditional probability of the emissions (spikes and/or choice) at each time bin. For time bins of each trial other than the last, it is the product of the conditional likelihood of all spike trains. For the last time bin, it corresponds to the product of the conditional likelihood of the spike trains and the choice. Element p𝐘𝑑[i][m][t][j,k] corresponds to ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k) across N neural units at the t-th time bin in the m-th trial of the i-th trialset. The last element p𝐘𝑑[i][m][end][j,k] of each trial corresponds to p(𝑑 | aₜ = ξⱼ, zₜ=k) ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k)"
	p𝐘𝑑::VVVMF
end
