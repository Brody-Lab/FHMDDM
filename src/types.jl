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
    Options

Model settings

"""
@with_kw struct Options{TB<:Bool,
						TS<:String,
						TF<:AbstractFloat,
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
	"whether to fit the left-right scaling factor"
	fit_a::TB=true
	"whether to fit the height of the sticky bounds"
	fit_B::TB=true
	"whether to fit the nonlinearity factor"
	fit_b::TB=true
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
	q_Aᶜ₂₂::TF= 1e-3; 	@assert q_Aᶜ₂₂ >= 0 && q_Aᶜ₂₂ <= 1
	"value of the bound height in native space that corresponds to zero in real space"
	q_B::TF=30.0; 		@assert q_B > 0
	"value of the adaptation change rate in native space that corresponds to zero in real space"
	q_k::TF=1e-3; 		@assert q_k > 0
	"value in native space of the sensitization strength parameter that corresponds to zero in real space"
	q_ϕ::TF=1-1e-3; 	@assert q_ϕ >= 0 && q_ϕ <= 1
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
	"lower bound of the initial noise"
	bound_σ²::TF = 1e-4
	"lower bound of the lapse rate"
	bound_ψ::TF = 1e-4
	"lower bound of the probabilities for the coupling variable"
	bound_z::TF = 1e-4
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
	SpikeTrainModel

The inputs and observations of a mixture of Poisson generalized linear model of a neuron's spike train for one trial
"""
@with_kw struct SpikeTrainModel{TVI<:Vector{<:Integer}, TMF<:Matrix{<:AbstractFloat}}
    "Columns of the design matrix that are invariant to the accumulator variable and correspond to regressors related to spike history or timing of events. Each column is scaled such that the maximum of the absolute value is 1."
    𝐔::TMF
    "Temporal bases values. Element 𝚽[t,i] corresponds to the value of the i-th temporal basis at the t-th time step"
    𝚽::TMF
    "response variable"
    𝐲::TVI
end

"""
    Trial

Information on the sensory stimulus and behavior each trial

Spike trains are not included. In sampled data, the generatives values of the latent variables are stored.
"""
@with_kw struct Trial{TB<:Bool,
                      TC<:Clicks,
                      TI<:Integer,
                      VI<:Vector{<:Integer},
					  VS<:Vector{<:SpikeTrainModel}}
    "information on the auditory clicks"
    clicks::TC
    "behavioral choice"
    choice::TB
    "number of time steps in this trial. The duration of each trial is from the onset of the stereoclick to the end of the fixation period"
    ntimesteps::TI
    "location of the reward baited in the previous trial (left:-1, right:1, no previous trial:0)"
    previousanswer::TI
	"vector of whose each element is a structure containing the input and observations of the mixture of poisson GLMs of the spike train of a neuron in this trial"
	spiketrainmodels::VS
    "generative values of the accumulator index variable (𝐚). This is nonempty only in sampled data"
    a::VI=Vector{Int}(undef,0)
    "generative values of the coupling variable (𝐜). This is nonempty only in sampled data"
    c::VI=Vector{Int}(undef,0)
end

"""
	GLMθ
"""
@with_kw struct GLMθ{TVR<:Vector{<:Real}}
	"regression coefficients of the accumulator-independent regressors"
    𝐮::TVR
    "Time-varying weighte. Element 𝐯[i] corresponds to the weight of the i-th temporal basis"
    𝐯::TVR
    "The exponent `e^a` specifies the ratio of right to left weight"
	a::TVR=zeros(eltype(𝐮),1)
	"Parameter specifying how the accumulator is nonlinearly transformed before inputted into the generalized linear model"
	b::TVR=zeros(eltype(𝐮),1)
end

"""
    MixturePoissonGLM

Mixture of Poisson generalized linear model
"""
@with_kw struct MixturePoissonGLM{TF<:AbstractFloat,
                                  TI<:Integer,
                                  TVF<:Vector{<:AbstractFloat},
								  TVI<:Vector{<:Integer},
								  Tθ<:GLMθ,
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
	"parameters"
	θ::Tθ
	"full design matrix"
	𝐗::TMF
	"Normalized values of the accumulator"
    𝛏::TVF
    "response variable"
    𝐲::TVI
    "factorial of the response variable"
    𝐲!::TVI = factorial.(𝐲)
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
	Model

A factorial hidden Markov drift-diffusion model
"""
@with_kw struct Model{Toptions<:Options,
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
					   VVG<:Vector{<:Vector{<:GLMθ}}}
	"parameters specifying the mixture of Poisson generalized linear model"
	glmθ::VVG
	"parameters specifying the latent variables"
	latentθ::L
end

"""
	Trialinvariant

A collection of hyperparameters and temporary quantities that are fixed across trials

"""
@with_kw struct Trialinvariant{	TI<:Integer,
								VF<:Vector{<:AbstractFloat},
								VR<:Vector{<:Real},
								MR<:Matrix{<:Real},
								MR2<:Matrix{<:Real},
							   	F<:AbstractFloat}
	"transition matrix of the accumulator variable in the absence of input"
	Aᵃsilent::MR
	"transition matrix of the coupling variable"
	Aᶜ::MR2=zeros(1,1)
	"transpose of the transition matrix of the coupling variable"
	Aᶜᵀ::MR
	"derivitive with respect to the means of the transition matrix of the accumulator variable in the absence of input"
	dAᵃsilentdμ::MR2=zeros(1,1)
	"derivitive with respect to the variance of the transition matrix of the accumulator variable in the absence of input"
	dAᵃsilentdσ²::MR2=zeros(1,1)
	"derivitive with respect to the bound of the transition matrix of the accumulator variable in the absence of input"
	dAᵃsilentdB::MR2=zeros(1,1)
	"time step, in seconds"
	Δt::F
	"an intermediate term used for computing the derivative with respect to the bound for the first time bin"
	𝛚::VF=zeros(1)
	"an intermediate term used for computing the derivative with respect to the bound for subsequent time bins"
	Ω::MR2=zeros(1,1)
	"prior probability of the coupling variable "
	πᶜᵀ::MR
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

"""
	CVIndices

Indices of trials and timesteps used for training and testing
"""
@with_kw struct CVIndices{VVI<:Vector{<:Vector{<:Integer}}}
	"`testingtrials[i]` indexes the trials from the i-th trialset used for testing"
	testingtrials::VVI
	"`trainingtrials[i]` indexes the trials from the i-th trialset used for training"
	trainingtrials::VVI
	"`testingtimesteps[i]` indexes the time steps from the i-th trialset used for testing"
	testingtimesteps::VVI
	"`trainingtimesteps[i]` indexes the time steps from the i-th trialset used for training"
	trainingtimesteps::VVI
end

"""
	CVResults

Results of cross-validation
"""
@with_kw struct CVResults{VC<:Vector{<:CVIndices},
							VL<:Vector{<:Latentθ},
							VVVG<:Vector{<:Vector{<:Vector{<:GLMθ}}},
							VVF<:Vector{<:Vector{<:AbstractFloat}},
							VF<:Vector{<:AbstractFloat}}
	"cvindices[k] indexes the trials and timesteps used for training and testing in the k-th resampling"
	cvindices::VC
	"θ₀native[k] specify the initial values of the parameters of the latent variables in the k-th resampling"
	θ₀native::VL
	"θnative[k] specify the optimized values of the parameters of the latent variables in the k-th resampling"
	θnative::VL
	"glmθ[k][i][n] specify the optimized values of the parameters of the n-th neuron's GLM in the i-th trialset in the k-th resampling"
	glmθ::VVVG
	"losses[k][i] specify the value of the loss function in the i-th iteration of the optimization in the k-th resampling"
	losses::VVF
	"gradientnorms[k][i] specify the 2-norm of gradient the loss function in the i-th iteration of the optimization in the k-th resampling"
	gradientnorms::VVF
	"rll_choice[i] indicate the trial-averaged log-likelihood of the choices in the i-th trialset, relative to the baseline trial-average log-likelihood computed under a Bernoulli distribution parametrized by fraction of right responses"
	rll_choice::VF
	"rll_spikes[i][n] indicate the time-averaged log-likelihood of the spike train of the n-th neuron in the -th trialset, relative to the baseline time-averaged log-likelihood computed under a Poisson distribution parametrized by mean spike train response"
	rll_spikes::VVF
end

"""
	Probabilityvector

First and second partial derivatives of a probability vector of the accumulator and quantities used for computing these derivatives
"""
@with_kw struct Probabilityvector{TI<:Integer,
								  TVI<:Vector{<:Integer},
								  TR<:Real,
								  TVR<:Vector{<:Real}}
	"------hyperparameters------"
	"duration of the time step"
	Δt::TR
	"number of discrete states of the accumulator"
	Ξ::TI

	"------parameters------"
	"parameter for bound height"
	B::TR
	"parameter for adptation change rate"
	k::TR
	"parameter for feedback"
	λ::TR
	"parameter for a constant offset in the mean in the prior probability"
	μ₀::TR
	"parameter for adaptation strength"
	ϕ::TR
	"parameter for the variance of the diffusion noise"
	σ²ₐ::TR
	"parameter for the variance of the prior probability"
	σ²ᵢ::TR
	"parameter for the variance of the per-click noise"
	σ²ₛ::TR
	"parameter for the weight of the previous reward's location"
	wₕ::TR

	"------intermediate quantities constant across time steps------"
	λΔt::TR = λ*Δt
	expλΔt::TR = exp(λΔt)
	"a vector representing the derivative of each discrete value of the accumulator with respect to the bound height"
	d𝛏_dB::TVR = (2collect(1:Ξ).-Ξ.-1)./(Ξ-2)
	"a vector representing discrete values of the accumulator"
	𝛏::TVR = B.*d𝛏_dB
	"spacing between consecutive discrete values of the accumulator"
	Δξ::TR = 𝛏[2]-𝛏[1]
	"derivative of the mean of a probability vector at one time step with respect to the differential auditory input (sum of the adapted magnitude from all right clicks, minus the summed adapted magnitudes from left clicks, for all clicks in the time step)"
	dμ_dΔc::TR = differentiate_μ_wrt_Δc(Δt, λ)
	"second derivative of the mean with respect to the differential auditory input and the feedback parameter "
	d²μ_dΔcdλ::TR = differentiate_μ_wrt_Δcλ(Δt, λ)
	"third derivative of the mean with respect to the differential auditory input and the feedback parameter twice"
	d³μ_dΔcdλdλ::TR = differentiate_μ_wrt_Δcλλ(Δt, λ)
	"derivative of the variance of a probability vector of a probability vector at one time step with respect to the aggregate auditory input (sum of the adapted magnitude from all right clicks, plus the summed adapted magnitudes from left clicks, for all clicks in the time step)"
	dσ²_d∑c::TR = σ²ₛ
	"a vector whose element `d²𝛍_dBdλ[j]` represents the derivative of the mean given that in the previous time step, the accumulator had the j-th discrete value, with respect to the bound height and the feedback parameters"
	d²𝛍_dBdλ::TVR = Δt.*expλΔt.*d𝛏_dB
	"location of the previous reward"
	previousanswer::TVI = zeros(Int,1)

	"------intermediate quantities updated at each time step------"
	"differential auditory input: sum of the adapted magnitude from all right clicks, minus the summed adapted magnitudes from left clicks, for all clicks in the time step"
	Δc::TVR = fill(NaN,1)
	"aggregate auditory input: sum of the adapted magnitude from all right clicks, plus the summed adapted magnitudes from left clicks, for all clicks in the time step"
 	∑c::TVR = fill(NaN,1)
	"derivative of the differential auditory input with respect to the adaptation change rate"
	dΔc_dk::TVR = fill(NaN,1)
	"derivative of the aggregate auditory input with respect to the adaptation change rate"
	d∑c_dk::TVR = fill(NaN,1)
	"derivative of the differential auditory input with respect to the adaptation strength"
	dΔc_dϕ::TVR = fill(NaN,1)
	"derivative of the aggregate auditory input with respect to the adaptation strength"
	d∑c_dϕ::TVR = fill(NaN,1)
	"second derivative of the differential auditory input with respect to the adaptation change rate"
	d²Δc_dkdk::TVR = fill(NaN,1)
	"second derivative of the aggregate auditory input with respect to the adaptation change rate"
	d²∑c_dkdk::TVR = fill(NaN,1)
	"second derivative of the differential auditory input with respect to the adaptation change rate and the adaptation strength"
	d²Δc_dkdϕ::TVR = fill(NaN,1)
	"second derivative of the aggregate auditory input with respect to the adaptation change rate and the adaptation strength"
	d²∑c_dkdϕ::TVR = fill(NaN,1)
	"second derivative of the differential auditory input with respect to the adaptation strength"
	d²Δc_dϕdϕ::TVR = fill(NaN,1)
	"second derivative of the aggregate auditory input with respect to the adaptation strength"
	d²∑c_dϕdϕ::TVR = fill(NaN,1)
	"variance of the probability vector"
	σ²::TVR = fill(NaN,1)
	"standard deviation of the probability vector"
	σ::TVR = fill(NaN,1)
	"standard deviation divided by the spacing between discrete values"
	σ_Δξ::TVR = fill(NaN,1)
	"standard deviation multiplied by the spacing between discrete values and by 2"
	σ2Δξ::TVR = fill(NaN,1)
	"variance multiplied by the spacing between discrete values and by 2"
	Δξσ²2::TVR = fill(NaN,1)
	"a vector whose j-th element represents the conditional mean of the probability vector, given that the accumulator in the previous time step being equal to the j-th discrete value"
	𝛍::TVR = fill(NaN,Ξ)
	"a vector of derivatives of the conditional means with respect to the feedback parameter"
	d𝛍_dλ::TVR = fill(NaN,Ξ)
	"a vector of second derivatives of the conditional means with respect to the feedback parameter"
	d²𝛍_dλdλ::TVR = fill(NaN,Ξ)
	"derivative of the mean with respect to the adaptation change rate"
	dμ_dk::TVR = fill(NaN,1)
	"derivative of the mean with respect to the adaptation strength"
	dμ_dϕ::TVR = fill(NaN,1)
	"second derivative of the mean with respect to the adaptation change rate"
	d²μ_dkdk::TVR = fill(NaN,1)
	"second derivative of the mean with respect to the adaptation change rate and adaptation strength"
	d²μ_dkdϕ::TVR = fill(NaN,1)
	"second derivative of the mean with respect to the adaptation strength"
	d²μ_dϕdϕ::TVR = fill(NaN,1)
	"derivative of the variance with respect to the adaptation change rate"
	dσ²_dk::TVR = fill(NaN,1)
	"derivative of the variance with respect to the adaptation strength"
	dσ²_dϕ::TVR = fill(NaN,1)
	"second derivative of the variance with respect to the adaptation change rate"
	d²σ²_dkdk::TVR = fill(NaN,1)
	"second derivative of the variance with respect to the adaptation change rate and adaptation strength"
	d²σ²_dkdϕ::TVR = fill(NaN,1)
	"second derivative of the variance with respect to the adaptation strength"
	d²σ²_dϕdϕ::TVR = fill(NaN,1)

	"------quantities updated at each time step and for each column of the transition matrix------"
	"z-scores computed using the discrete value of the accumulator, the mean, and the standard deviation"
	𝐳::TVR =  fill(NaN,Ξ)
	"normal probability density function evaluated at each z-score"
	𝐟::TVR =  fill(NaN,Ξ)
	"quantities used for computing derivatives with respect to bound height"
	𝛈::TVR =  fill(NaN,Ξ)
	"quantities used for computing derivatives with respect to bound height"
	𝛚::TVR =  fill(NaN,Ξ)
	"normal cumulative distibution function evaluated at each z-score"
	Φ::TVR =  fill(NaN,Ξ)
	"normal complementary cumulative distibution function evaluated at each z-score"
	Ψ::TVR =  fill(NaN,Ξ)
	"difference between the normal probability density function evaluated at succesive z-scores"
	Δf::TVR = fill(NaN,Ξ-1)
	"difference between the normal standardized distribution function evaluated at succesive z-scores"
	ΔΦ::TVR = fill(NaN,Ξ-1)
	"values of the probability vector"
	𝛑::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the bound height"
	d𝛑_dB::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the mean"
	d𝛑_dμ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the variance"
	d𝛑_dσ²::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the adaptation change rate"
	d𝛑_dk::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the feedback"
	d𝛑_dλ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the constant offset to the mean of the prior probability"
	d𝛑_dμ₀::TVR = d𝛑_dμ
	"derivative of the probability vector with respect to the adaptation strength"
	d𝛑_dϕ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the variance of the diffusion noise"
	d𝛑_dσ²ₐ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the variance of the prior probability"
	d𝛑_dσ²ᵢ::TVR = d𝛑_dσ²
	"derivative of the probability vector with respect to the variance of the per-click noise"
	d𝛑_dσ²ₛ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the weight of the location of the previous reward"
	d𝛑_dwₕ::TVR = fill(NaN,Ξ)
	"second derivative of the probability vector with respect to the bound height"
	d²𝛑_dBdB::TVR = fill(NaN,Ξ)
	"second derivative of the probability vector with respect to the bound height and mean"
	d²𝛑_dBdμ::TVR = fill(NaN,Ξ)
	"second derivative of the probability vector with respect to the bound height and variance"
	d²𝛑_dBdσ²::TVR = fill(NaN,Ξ)
	"second derivative of the probability vector with respect to the mean"
	d²𝛑_dμdμ::TVR = fill(NaN,Ξ)
	"second derivative of the probability vector with respect to the mean and variance"
	d²𝛑_dμdσ²::TVR = fill(NaN,Ξ)
	"second derivative of the probability vector with respect to the variance"
	d²𝛑_dσ²dσ²::TVR = fill(NaN,Ξ)
	"second derivative of the probability vector with respect to the bound height and the constant offset to the mean of the prior probability"
	d²𝛑_dBdμ₀::TVR = d²𝛑_dBdμ
	"derivative of the probability vector with respect to the bound height and the variance of the prior probability"
	d²𝛑_dBdσ²ᵢ::TVR = d²𝛑_dBdσ²
	"derivative of the probability vector with respect to the bound height and the weight of the location of the previous reward"
	d²𝛑_dBdwₕ::TVR = fill(NaN,Ξ)
	"second derivative of the probability vector with respect to the constant offset to the mean of the prior probability"
	d²𝛑_dμ₀dμ₀::TVR = d²𝛑_dμdμ
	"second derivative of the probability vector with respect to the constant offset to the mean of the prior probability and the variance of the prior probability"
	d²𝛑_dμ₀dσ²ᵢ::TVR = d²𝛑_dμdσ²
	"second derivative of the probability vector with respect to the constant offset to the mean of the prior probability and the weight of the location of the previous reward"
	d²𝛑_dμ₀dwₕ::TVR = fill(NaN,Ξ)
	"second derivative of the probability vector with respect to the variance of the prior probability"
	d²𝛑_dσ²ᵢdσ²ᵢ::TVR = d²𝛑_dσ²dσ²
	"second derivative of the probability vector with respect to the variance of the prior probability and the weight of the location of the previous reward"
	d²𝛑_dσ²ᵢdwₕ::TVR = fill(NaN,Ξ)
	"second derivative of the probability vector with to the weight of the location of the previous reward"
	d²𝛑_dwₕdwₕ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the bound height and the adaptation change rate"
	d²𝛑_dBdk::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the bound height and the feedback"
	d²𝛑_dBdλ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the bound height and the adaptation strength"
	d²𝛑_dBdϕ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the bound height and the variance of the diffusion noise"
	d²𝛑_dBdσ²ₐ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the bound height and the variance of the per-click noise"
	d²𝛑_dBdσ²ₛ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the adaptation change rate"
	d²𝛑_dkdk::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the adaptation change rate and the feedback"
	d²𝛑_dkdλ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the adaptation change rate and adaptation strength"
	d²𝛑_dkdϕ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the adaptation change rate and variance of the diffusion noise"
	d²𝛑_dkdσ²ₐ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the adaptation change rate and variance of the per-click noise"
	d²𝛑_dkdσ²ₛ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the feedback strength"
	d²𝛑_dλdλ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the feedback strength and adaptation strength"
	d²𝛑_dλdϕ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the feedback strength and variance of the diffusion noise"
	d²𝛑_dλdσ²ₐ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the feedback strength and variance of the per-click noise"
	d²𝛑_dλdσ²ₛ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the adaptation strength"
	d²𝛑_dϕdϕ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the adaptation strength and variance of the diffusion noise"
	d²𝛑_dϕdσ²ₐ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the adaptation strength and variance of the per-click noise"
	d²𝛑_dϕdσ²ₛ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the variance of the diffusion noise"
	d²𝛑_dσ²ₐdσ²ₐ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the variance of the diffusion noise and the variance of the per-click noise"
	d²𝛑_dσ²ₐdσ²ₛ::TVR = fill(NaN,Ξ)
	"derivative of the probability vector with respect to the variance of the per-click noise"
	d²𝛑_dσ²ₛdσ²ₛ::TVR = fill(NaN,Ξ)
end

"""
	Adaptedclicks

The post-adaptation magnitude of each click and the first- and second-order partial derivatives of the post-adaptation magnitude
"""
@with_kw struct Adaptedclicks{TVR1<:Vector{<:Real}, TVR2<:Vector{<:Real}}
	"adapted strengths of the clicks"
	C::TVR1
	"derivative of adapted click strengths with respect to the adaptation change rate"
	dC_dk::TVR2=zeros(0)
	"derivative of adapted click strengths with respect to the adaptation strength"
	dC_dϕ::TVR2=zeros(0)
	"second derivative of adapted click strengths with respect to the adaptation change rate"
	d²C_dkdk::TVR2=zeros(0)
	"second derivative of adapted click strengths with respect to the adaptation change rate and adaptation strength"
	d²C_dkdϕ::TVR2=zeros(0)
	"second derivative of adapted click strengths with respect to the adaptation strength"
	d²C_dϕdϕ::TVR2=zeros(0)
end
