"""
    Options

Model settings
"""
@with_kw struct Options{TB<:Bool, TS<:String, TF<:AbstractFloat, TI<:Integer, TVF<:Vector{<:AbstractFloat}}
	"response latency of the accumulator to the clicks"
    a_latency_s::TF=1e-2
	"value optimized when initializing the choice-related parameters"
	choiceobjective::TS="posterior"
	"Exponent used to compute the scale factor of the log-likelihood of the choices. The scale factor is computed by raising the product of the number of neurons and the average number of time steps in each trial to the exponent. An exponent of 0 means no scaling"
	choiceLL_scaling_exponent::TF=0.6
	"full path of the data"
    datapath::TS=""
	"duration of each timestep in seconds"
    Δt::TF=0.01
	"whether to fit the height of the sticky bounds"
	fit_B::TB=true
	"whether to fit the parameter for transforming the accumulator"
	fit_b::TB=false
	"whether to fit the parameter for computing the coupling probability"
	fit_c::TB=true
	"whether to fit separate encoding weights for when the accumulator is at the bound"
	fit_β::TB=true
	"whether to fit the exponential change rate of inter-click adaptation"
	fit_k::TB=false
	"whether to fit the parameter specifying leak or instability"
	fit_λ::TB=false
	"whether to fit the constant added to the mean of the distribution of the accumulator variable at the first time step"
	fit_μ₀::TB=true
	"whether to fit the strength of inter-click adaptation and sign of the adaptation (facilitation vs. depression)"
	fit_ϕ::TB=false
	"whether to fit the behavioral lapse rate"
	fit_ψ::TB=false
	"whether to fit the variance of the Gaussian noise added at each time step"
	fit_σ²ₐ::TB=false
	"whether to fit the variance of the prior probability of the accumulator variable"
	fit_σ²ᵢ::TB=false
	"whether to fit the variance of Gaussian noise added as a result of the clicks"
	fit_σ²ₛ::TB=true
	"whether to fit the weight of the rewarded option of the previous trial on the mean of the accumulator at the first time step"
	fit_wₕ::TB=false
	"L2 norm of the gradient at which convergence of model's cost function is considered to have converged"
	g_tol::TF=1e-8
	"maximum and minimum of the L2 shrinkage penalty for each class of parameters. The penalty is initialized as (and if not being learned, set as) the geometric mean of the maximum and minimum."
	"accumulator transformation"
	L2_b_max::TF=1e-2
	L2_b_min::TF=1e-4
	"couplng probability"
	L2_c_max::TF=1e-2
	L2_c_min::TF=1e-4
	"latent variable parameter when fitting to only choices"
	L2_choices_max::TF=1e0
	L2_choices_min::TF=1e-4
	"gain"
	L2_gain_max::TF=1e-5
	L2_gain_min::TF=1e-7
	"spike history"
	L2_postspike_max::TF=1e-2
	L2_postspike_min::TF=1e-4
	"pre-movement"
	L2_premovement_max::TF=1e-2
	L2_premovement_min::TF=1e-4
	"post-photostimulus filter"
	L2_postphotostimulus_max::TF=1e-2
	L2_postphotostimulus_min::TF=1e-4
	"post-stereoclick filter"
	L2_poststereoclick_max::TF=1e-2
	L2_poststereoclick_min::TF=1e-4
	"latent variable parameter"
	L2_latent_max::TF=10.0
	L2_latent_min::TF=0.1
	"encoding of accumulated evidence"
	L2_v_max::TF=1e-4
	L2_v_min::TF=1e-6
	"bound height"
	B_l::TF=10.0
	B_q::TF=15.0
	B_u::TF=20.0
	"adaptation change rate"
	k_l::TF=1e-4
	k_q::TF=1e-3
	k_u::TF=1e3
	"feedback"
	λ_l::TF = -5.0
	λ_q::TF = 0.0
	λ_u::TF = 5.0
	"bias"
	μ₀_l::TF=-5.0
	μ₀_q::TF= 0.0
	μ₀_u::TF= 5.0
	"adaptation strength"
	ϕ_l::TF=1e-4
	ϕ_q::TF=1-1e-3
	ϕ_u::TF=1.0-1e-4
	"behavioral lapse rate. A value of 0 will result in underflow"
	ψ_l::TF=1e-4
	ψ_q::TF=1e-2
	ψ_u::TF=1-1e-4
	"variance of the diffusion noise (i.e., zero-meaned, independent and identically distributed gaussian noise added at each time step)"
	σ²ₐ_l::TF=1e-4
	σ²ₐ_q::TF=1e-3
	σ²ₐ_u::TF=1e-2
	"variance of the initial probability of the accumulator variable"
	σ²ᵢ_l::TF=1e-1
	σ²ᵢ_q::TF=1e0
	σ²ᵢ_u::TF=1e1
	"variance of the variance of per-click noise"
	σ²ₛ_l::TF = 0.1
	σ²ₛ_q::TF = 5.0
	σ²ₛ_u::TF = 20.0
	"weight of previous answer"
	wₕ_l::TF = -5.0
	wₕ_q::TF = 0.0
	wₕ_u::TF = 5.0
	"minimum value of the prior and transition probabilities of the accumulator"
	minpa::TF=1e-8
	"value to maximized to learn the parameters"
	objective::TS="posterior"; @assert any(objective .== ["evidence", "posterior", "likelihood", "initialization"])
	"absolute path of the folder where the model output, including the summary and predictions, are saved"
	outputpath::TS=""
	"coefficient multiplied to any scale factor of a parameter of a Poisson mixture GLM"
	sf_mpGLM::TVF=[NaN]
    "scale factor of the conditional likelihood of the spiking of a neuron at a time step"
	sf_y::TF=1.2
	"maximum number of basis functions"
	tbf_gain_maxfunctions::TI=8
	"Options for the temporal basis function whose linear combination constitute the post-spike filter. The setting `tbf_postspike_dur_s` is the duration, in seconds, of the filter. The setting `tbf_postspike_linear` determines whether a linear function is included in the basis."
	tbf_postspike_begins0::TB=false
	tbf_postspike_dur_s::TF=0.25
	tbf_postspike_ends0::TB=true
	tbf_postspike_hz::TF=12.0
	tbf_postspike_stretch::TF=1.0
	"Options for the temporal basis associated with the pre-movement filter"
	tbf_premovement_begins0::TB=true
	tbf_premovement_dur_s::TF=0.6
	tbf_premovement_ends0::TB=false
	tbf_premovement_hz::TF=3.0
	tbf_premovement_stretch::TF=0.1
	"Options for the temporal basis associated with the post-photostimulus filter"
	tbf_postphotostimulus_begins0::TB=false
	tbf_postphotostimulus_ends0::TB=false
	tbf_postphotostimulus_hz::TF=NaN
	tbf_postphotostimulus_stretch::TF=1.0
	"Options for the temporal basis associated with the post-stereoclick filter"
	tbf_poststereoclick_begins0::TB=false
	tbf_poststereoclick_dur_s::TF=1.0
	tbf_poststereoclick_ends0::TB=true
	tbf_poststereoclick_hz::TF=5.0
	tbf_poststereoclick_stretch::TF=0.2
    "number of states of the discrete accumulator variable"
    Ξ::TI=53; @assert isodd(Ξ) && Ξ > 1
end

"""
	Latentθ

Parameters of the latent variables in the factorial hidden Markov drift-diffusion model.

Not included are the weights of the linear filters of the mixture of Poisson generalized linear model of each neuron
"""
@with_kw struct Latentθ{VR<:Vector{<:Real}}
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
@with_kw struct Trial{TB<:Bool, TC<:Clicks, TF<:AbstractFloat, TI<:Integer, TVVI<:Vector{<:Vector{<:Integer}}}
    "information on the auditory clicks"
    clicks::TC
    "behavioral choice"
    choice::TB
	"log of the ratio of the generative right and left click rate"
	γ::TF
	"index of the trial in the trialset"
	index_in_trialset::TI
	"time of leaving the center port, relative to the time of the stereoclick, in seconds"
	movementtime_s::TF; @assert movementtime_s > 0
	"time of leaving the center port, relative to the time of the stereoclick, in time steps"
	movementtimestep::TI; @assert movementtimestep > 0
    "number of time steps in this trial. The duration of each trial is from the onset of the stereoclick to the end of the fixation period"
    ntimesteps::TI
	"time when the offset ramp of the photostimulus began"
	photostimulus_decline_on_s::TF
	"time when the onset ramp of the photostimulus began"
	photostimulus_incline_on_s::TF
    "location of the reward baited in the previous trial (left:-1, right:1, no previous trial:0)"
    previousanswer::TI
	"a nested array whose element `spiketrains[n][t]` is the spike train response of the n-th neuron on the t-th time step of the trial"
	spiketrains::TVVI
	"time of the stereoclick, in seconds, in the sessions"
	stereoclick_time_s::TF
	"number of timesteps in the trialset preceding this trial"
	τ₀::TI
    "index of the trialset to which this trial belongs"
	trialsetindex::TI
end

"""
	Indices𝐮

Indices of the encoding weights of the temporal basis vectors of the filters that are independent of the accumulator
"""
@with_kw struct Indices𝐮{UI<:UnitRange{<:Integer}}
	gain::UI=1:1
	postspike::UI
	poststereoclick::UI
	premovement::UI
	postphotostimulus::UI
end

"""
	GLMθ

Parameters of a mixture of Poisson generalized linear model
"""
@with_kw struct GLMθ{B<:Bool, IU<:Indices𝐮, VR<:Vector{<:Real}, VS<:Vector{<:Symbol}}
    "nonlinearity in accumulator transformation"
	b::VR=[0.0]
	"coupling parameter"
	c::VR=[0.0]
	"order by which parameters are concatenated"
	concatenationorder::VS = [:𝐮, :v, :β, :b, :c]
	"whether the nonlinearity parameter is fit"
	fit_b::B
	"whether the coupling parameter is fit"
	fit_c::B
	"whether to fit separate encoding weights for when the accumulator at the bound"
	fit_β::B
	"state-independent linear filter of inputs from the spike history and time in the trial"
    𝐮::VR
	"Indices of the encoding weights of the temporal basis vectors of the filters that are independent of the accumulator"
	indices𝐮::IU
    "state-dependent linear filters of the inputs from the accumulator "
    v::VR=[NaN]
	"state-dependent linear filters of the time-varying input from the transformed accumulated evidence"
	β::VR=[NaN]
end

"""
    MixturePoissonGLM

Mixture of Poisson generalized linear model
"""
@with_kw struct MixturePoissonGLM{TI<:Integer,
								  F<:AbstractFloat,
								  UI<:UnitRange{<:Integer},
                                  VF<:Vector{<:AbstractFloat},
								  VI<:Vector{<:UInt8},
								  Tθ<:GLMθ,
                                  MF<:Matrix{<:AbstractFloat}}
    "size of the time bin"
    Δt::F
	"Normalized values of the accumulator"
    d𝛏_dB::VF
	"values of the basis functions parametrizing the slow drift in gain on each trial"
	Φgain::MF
	"Values of the smooth temporal basis functions used to parametrize the post-spike filter"
	Φpostspike::MF
	"Values of the smooth temporal basis functions used to parametrize the time-varying relationship between the timing of the animal leaving the center and the neuron's probability of spiking. The timing is represented by a delta function, and the delta function is convolved with a linear combination of the temporal basis functions to specify the filter, or the kernel, of the event. The columns correspond to temporal basis functions and rows correspond to time steps, concatenated across trials."
	Φpremovement::MF
	"temporal basis vectors for the photostimulus"
	Φpostphotostimulus::MF
	"time steps of the temporal basis vectors relative to the onset of the photostimulus"
	Φpostphotostimulus_timesteps::UI
	"Values of the smooth temporal basis functions used to parametrize the time-varying relationship between the timing of the stereoclick and the neuron's probability of spiking."
	Φpoststereoclick::MF
	"scale factor used for all parameter of the GLM"
	sf_mpGLM::F
	"scale factor multiplied to the likelihood to avoid underflow when computing the likelihood of the population response"
	sf_y::F
	"parameters"
	θ::Tθ
	"design matrix. The first column are ones. The subsequent columns correspond to spike history-dependent inputs. These are followed by columns corresponding to the time-dependent input. The last set of columns are given by 𝐕"
	𝐗::MF
	"number of accumulator states"
	Ξ::TI=length(d𝛏_dB)
	"index of the state for which the accumulator is 0"
	index0::TI=cld(Ξ,2)
	"Poisson observations"
    𝐲::VI
end

"""
    Trialset

A group of trials in which a population of neurons were recorded simultaneously
"""
@with_kw struct Trialset{VM<:Vector{<:MixturePoissonGLM}, TI<:Integer, VT<:Vector{<:Trial}}
	"Mixture of Poisson GLM of each neuron in this trialset"
    mpGLMs::VM=MixturePoissonGLM[]
	"number of time steps summed across trials"
	ntimesteps::TI=size(mpGLMs[1].𝐗,1)
	"Information on the stimulus and behavior for each trial in this trial-set"
    trials::VT
	"Number of trials"
	ntrials::TI=length(trials)
end

"""
	GaussianPrior

Information on the zero-meaned Gaussian prior distribution on the values of the parameters in real space
"""
@with_kw struct GaussianPrior{	VI<:Vector{<:Integer},
								VF<:Vector{<:AbstractFloat},
								VR<:Vector{<:Real},
								VS<:Vector{<:String},
								MR<:Matrix{<:Real},
								VVI<:Vector{<:Vector{<:Integer}},
								VMF<:Vector{<:Matrix{<:AbstractFloat}}}
	"L2 penalty matrices"
	𝐀::VMF
	"L2 penalty coefficients"
	𝛂::VR
	"minimum values of the L2 penalty coefficients"
	𝛂min::VF
	"maximum values of the L2 penalty coefficients"
	𝛂max::VF
	"Indices of the parameters related to each L2 penalty coefficient: element `index𝐀[i][j]` corresponds to the i-th group of parameters and the j-th parameter in that group"
	index𝐀::VVI
	"the precision matrix, i.e., inverse of the covariance matrix, of the gaussian prior on the model parameters"
	𝚲::MR
	"indices of the dimensions with finite variance"
	index𝚽::VI = sort(union(index𝐀...))
	"square submatrix of the precision matrix after deleting the columns and rows corresponding to the dimensions with infinite variance"
	𝚽::MR= 𝚲[index𝚽,index𝚽]
	"indices of 𝐀 within `index𝚽`"
	index𝐀_in_index𝚽::VVI = map(indexA->map(indexAᵢⱼ->findfirst(index𝚽.==indexAᵢⱼ), indexA), index𝐀)
	"the name of each L2 penalty"
	penaltynames::VS
end

"""
	Model

A factorial hidden Markov drift-diffusion model
"""
@with_kw struct Model{Toptions<:Options,
					GP<:GaussianPrior,
					Tθ1<:Latentθ,
					Tθ2<:Latentθ,
					Tθ3<:Latentθ,
					VT<:Vector{<:Trialset}}
	"settings of the model"
	options::Toptions
	"Gaussian prior on the parameters"
	gaussianprior::GP
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
@with_kw struct Indexθ{L<:Latentθ, VVG<:Vector{<:Vector{<:GLMθ}}}
	"parameters specifying the mixture of Poisson generalized linear model"
	glmθ::VVG
	"parameters specifying the latent variables"
	latentθ::L
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
	"minimum value of the accumulator prior or transition probability"
	minpa::TR
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
	"1.0 - Ξ*minpa"
	one_minus_Ξminpa::TR = 1.0 - Ξ*minpa

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

"""
	Sameacrosstrials

Quantities that are same across trials and used in each trial
"""
@with_kw struct Sameacrosstrials{MMR<:Matrix{<:Matrix{<:Real}},
								VMR<:Vector{<:Matrix{<:Real}},
								VTMR<:Vector{<:Transpose{<:Real, <:Matrix{<:Real}}},
								MR<:Matrix{<:Real},
								TMR<:Transpose{<:Real, <:Matrix{<:Real}},
								VR<:Vector{<:Real},
								VVR<:Vector{<:Vector{<:Real}},
								TVR<:Transpose{<:Real, <:Vector{<:Real}},
								VTVR<:Vector{<:Transpose{<:Real, <:Vector{<:Real}}},
								R<:Real,
								VI<:Vector{<:Integer},
								VVI<:Vector{<:Vector{<:Integer}},
								VVVI<:Vector{<:Vector{<:Vector{<:Integer}}},
								TI<:Integer}
	"transition matrix of the accumulator at a time step without auditory input. Element `Aᵃsilent[q][i,j]` corresponds to the transition probability p{a(t)=ξ(i) ∣ a(t-1) = ξ(j)}"
	Aᵃsilent::MR
	"first-order partial derivatives of the transition matrix of the accumulator at a time step without auditory input. Element `∇Aᵃsilent[q][i,j]` corresponds to the derivative of the transition probability p{a(t)=ξ(i) ∣ a(t-1) = ξ(j)} with respect to the q-th parameter that influence the accumulator transitions."
	∇Aᵃsilent::VMR
	"second-order partial derivatives of the transition matrix of the accumulator at a time step without auditory input. Element `∇∇Aᵃsilent[q,r][i,j]` corresponds to the derivative of the transition probability p{a(t)=ξ(i) ∣ a(t-1) = ξ(j)} with respect to the q-th parameter and r-th parameter that influence the accumulator transitions."
	∇∇Aᵃsilent::MMR
	"transition matrix of the coupling"
	Aᶜ::MR
	"transpose of the transition matrix of the coupling. Element Aᶜᵀ[i,j] corresponds to the transition probability p{c(t)=j ∣ c(t-1)=i}"
	Aᶜᵀ::TMR=transpose(Aᶜ)
	"first-order partial derivatives of the transition matrix of the coupling. Element ∇Aᶜ[q][i,j] corresponds to the derivative of the transition probability p{c(t)=i ∣ c(t-1)=j} with respect to the q-th parameter that influence coupling transitions."
	∇Aᶜ::VMR
	"first-order partial derivatives of the transpose of the transition matrix of the coupling. Element ∇Aᶜᵀ[q][i,j] corresponds to the derivative of the transition probability p{c(t)=j ∣ c(t-1)=i} with respect to the q-th parameter that influence coupling transitions."
	∇Aᶜᵀ::VTMR = transpose.(∇Aᶜ)
	"size of the time step"
	Δt::R
	"indices of the parameters that influence the prior probabilities of the accumulator"
	indexθ_pa₁::VI
	"indices of the parameters that influence the transition probabilities of the accumulator"
	indexθ_paₜaₜ₋₁::VI
	"indices of the parameters that influence the transition probabilities of the accumulator"
	indexθ_paₜaₜ₋₁only::VI = setdiff(indexθ_paₜaₜ₋₁, indexθ_pa₁)
	"indices of the parameters that influence the prior probabilities of the coupling"
	indexθ_pc₁::VI
	"indices of the parameters that influence the transition probabilities of the coupling variable"
	indexθ_pcₜcₜ₋₁::VI
	"indices of the parameters that influence the lapse rate"
	indexθ_ψ::VI
	"indices of the parameters in each Poisson mixture GLM in each trialset"
	indexθ_py::VVVI
	"indices of the parameters in the Poisson mixture GLM in each trialset"
	indexθ_pY::VVI
	"indices of the parameters in each trialset"
	indexθ_trialset::VVI = map(indexθ_pY->vcat(1:13, indexθ_pY), indexθ_pY)
	"number of coupling states"
	K::TI
	"prior probability of the coupling"
	πᶜ::VR
	"transpose of the prior probability of the coupling. It is a row vector"
	πᶜᵀ::TVR=transpose(πᶜ)
	"first-order partial derivatives of the prior probability of the coupling. Element ∇πᶜ[q][i] corresponds to the derivative of prior probability p{c(t=1)=i} with respect to the q-th parameter that influence the prior probability of coupling."
	∇πᶜ::VVR
	"first-order partial derivatives of the transpose of the prior probability of the coupling."
	∇πᶜᵀ::VTVR=transpose.(∇πᶜ)
	"number of accumulator states"
	Ξ::TI
	"number of parameters that influence the prior probabilities of the accumulator"
	nθ_pa₁::TI = length(indexθ_pa₁)
	"number of parameters that influence the transition probabilities of the accumulator"
	nθ_paₜaₜ₋₁::TI = length(indexθ_paₜaₜ₋₁)
	"number of parameters that influence the prior probabilities of the coupling"
	nθ_pc₁::TI = length(indexθ_pc₁)
	"number of parameters that influence the transition probabilities of the coupling variable"
	nθ_pcₜcₜ₋₁::TI = length(indexθ_pcₜcₜ₋₁)
	"number of the parameters that influence the lapse rate"
	nθ_ψ::TI = length(indexθ_ψ)
	"number of parameters in the Poisson mixture GLM in each trialset"
	nθ_py::VVI = map(x->length.(x), indexθ_py)
	"number of parameters in the Poisson mixture GLM in each trialset"
	nθ_pY::VI = map(indices->length(indices), indexθ_pY)
	"total number of parameters in the model, including those not being fit"
	nθ_trialset::VI = length.(indexθ_trialset)
	"total number of parameters in the model, including those not being fit"
	nθ_alltrialsets::TI = sum(nθ_trialset)
	"whether a parameter influences the prior probability of the accumulator, and if so, the index of that parameter"
	index_pa₁_in_θ::VI = let x = zeros(Int, nθ_alltrialsets); x[indexθ_pa₁] .= 1:nθ_pa₁; x; end
	"whether a parameter influences the transition probability of the accumulator, and if so, the index of that parameter"
	index_paₜaₜ₋₁_in_θ::VI = let x = zeros(Int, nθ_alltrialsets); x[indexθ_paₜaₜ₋₁] .= 1:nθ_paₜaₜ₋₁; x; end
	"whether a parameter influences the prior probability of the coupling, and if so, the index of that parameter"
	index_pc₁_in_θ::VI = let x = zeros(Int, nθ_alltrialsets); x[indexθ_pc₁] .= 1:nθ_pc₁; x; end
	"whether a parameter influences the transition probability of the coupling, and if so, the index of that parameter"
	index_pcₜcₜ₋₁_in_θ::VI = let x = zeros(Int, nθ_alltrialsets); x[indexθ_pcₜcₜ₋₁] .= 1:nθ_pcₜcₜ₋₁; x; end
	"whether a parameter influences the prior probability of the lapse, and if so, the index of that parameter"
	index_ψ_in_θ::VI = let x = zeros(Int, nθ_alltrialsets); x[indexθ_ψ] .= 1:nθ_ψ; x; end
	"whether a parameter influences the mixture of Poisson GLM, and if so, the index of that parameter"
	index_pY_in_θ::VVI = map(indexθ_pY) do indices
							x = zeros(Int, nθ_alltrialsets)
							x[indices] .= 1:length(indices)
							x
						 end
end

"""
	Memoryforhessian

Pre-allocated memory for computing the hessian as the jacobian of the expectation conjugate gradient
"""
@with_kw struct Memoryforhessian{VR<:Vector{<:Real},
								MR<:Matrix{<:Real},
								VVR<:Vector{<:Vector{<:Real}},
								VVT<:Vector{<:Vector{<:GLMθ}},
								VMR<:Vector{<:Matrix{<:Real}},
								MVR<:Matrix{<:Vector{<:Real}},
								VVVR<:Vector{<:Vector{<:Vector{<:Real}}},
								VVMR<:Vector{<:Vector{<:Matrix{<:Real}}},
								VMMR<:Vector{<:Matrix{<:Matrix{<:Real}}},
								VVVMR<:Vector{<:Vector{<:Vector{<:Matrix{<:Real}}}},
								VVMMR<:Vector{<:Vector{<:Matrix{<:Matrix{<:Real}}}},
								VVMVR<:Vector{<:Vector{<:Matrix{<:Vector{<:Real}}}},
								PT<:Probabilityvector}
	"log-likelihood"
	ℓ::VR = zeros(1)
	"gradient of the log-likelihood"
	∇ℓ::VR
	"hessian of the log-likelihood"
	∇∇ℓ::MR
	"transition matrix of the accumulator at a time-step when there is input. Element `Aᵃinput[t][i,j]` corresponds to the t-th time step in a trial with input, i-th accumulator step in the current time step, and j-th accumulator state in the previous time step "
	Aᵃinput::VMR
	"partial derivatives of the transition matrix of the accumulator at a time-step when there is input. Element `∇Aᵃinput[q][t][i,j]` corresponds to the q-th drift-diffusion parameter, t-th time step in a trial with input, i-th accumulator step in the current time step, and j-th accumulator state in the previous time step "
	∇Aᵃinput::VVMR
	"second order partial derivatives of the transition matrix of the accumulator at a time-step when there is input. Element `∇∇Aᵃinput[q,r][t][i,j]` corresponds to the q-th and r-th drift-diffusion parameter, t-th time step in a trial with input, i-th accumulator step in the current time step, and j-th accumulator state in the previous time step "
	∇∇Aᵃinput::VMMR
	"past-conditioned likelihood. Element `D[t]` corresponds to the t-th time step of a trial"
	D::VR
	"gradient of the past-conditioned likelihood. Element `∇D[t][q]` corresponds to the t-th time step of a trial and q-th parameter among all parameters in the model"
	∇D::VVR
	"indices of the GLM parameters for each neuron in each trialset"
	indexθglms::VVT
	"derivative of the conditional likelihood of the emissions at the last time step of a trial with respect to the lapse parameter ψ. Element `∂pY𝑑_∂ψ[i,j]` corresponds to the i-th accumulator state and j-th coupling state."
	∂pY𝑑_∂ψ::MR
	"forward term. Element 'f[t][i,j]' corresponds to the t-th time step in a trial, i-th accumulator state, and j-th coupling state"
	f::VMR
	"gradient of the forward term. Element '∇f[t][q][i,j]' corresponds to the t-th time step in a trial, q-th parameter among all parameters in the model, i-th accumulator state, and j-th coupling state"
	∇f::VVMR
	"gradient of the backward term. Element '∇b[q][i,j]' corresponds to the q-th parameter among all parameters in the model, i-th accumulator state, and j-th coupling state"
	∇b::VMR
	"linear predictor"
	𝐋::VVMVR
	"conditional Poisson rate of each neuron at each time step of a trial. Element `λ[n][t][i,j]` corresponds to the n-th neuron in a trialset, t-th time step in a trial, i-th accumulator state, and j-th coupling state"
	λ::VVMR
	"first-order partial derivatives of the log-likelihood of the spiking of each neuron at each time step. Element '∇logpy[t][n][q][i,j]' corresponds to t-th time step in a trial, n-th neuron in a trialset, q-th parameter of that neuron's GLM, i-th accumulator state, and j-th coupling state"
	∇logpy::VVVMR
	"second-order partial derivatives of the log-likelihood of the spiking of each neuron at each time step. Element '∇∇logpy[t][n][q,r][i,j]' corresponds to t-th time step in a trial, n-th neuron in a trialset, q-th and r-th parameter of that neuron's GLM, i-th accumulator state, and j-th coupling state"
	∇∇logpy::VVMMR
	"first-order partial derivatives of the prior probability of the accumulator. Element `∇pa₁[q][i]` corresponds to the q-th parameter among the parameters that govern prior probability and i-th accumulator state"
	∇pa₁::VVR
	"second-order partial derivatives of the prior probability of the accumulator. Element `∇∇pa₁[q,r][i]` corresponds to the q-th and r-th parameter among the parameters that govern prior probability and i-th accumulator state"
	∇∇pa₁::MVR
	"overdispersion parameters of each neuron. Element 𝛂[i][n] corresponds to the n-th neuron in the i-th trialset."
	𝛂::VVR
	"transformed values of accumulated evidence. Element `𝛚[i][n][j]` corresponds to the transformation of the j-th discrete value of accumulated for the n-th neuron in the i-th trialset."
	𝛚::VVVR
	"first-order derivative of the transformed values of accumulated evidence"
	d𝛚_db::VVVR
	"second-order derivative of the transformed values of accumulated evidence"
	d²𝛚_db²::VVVR
	"condition likelihood of all emissions at a time step. Element `pY[t][i,j]` corresponds to the t-th time step in a trial, i-th accumulator state, and j-th coupling state"
	pY::VMR
	"first-order partial derivatives condition likelihood of all emissions at a time step. Element `∇pY[t][q][i,j]` corresponds to the t-th time step in a trial, q-th parameter among all parameters of all GLM's in a trialset (not including the lapse), i-th accumulator state, and j-th coupling state"
	∇pY::VVMR
	"`Probabilityvector`: a structure containing memory for computing the probability vector of the accumulator and the first- and second-order partial derivatives of the elements of the probability vector"
	P::PT
end

"""
	Memoryforgradient

Container of variables used by both the log-likelihood and gradient computation
"""
@with_kw struct Memoryforgradient{R<:Real,
								TI<:Integer,
								VI<:Vector{<:Integer},
								VR<:Vector{<:Real},
								MR<:Matrix{<:Real},
								VVR<:Vector{<:Vector{<:Real}},
								VVθ<:Vector{<:Vector{<:GLMθ}},
								VMR<:Vector{<:Matrix{<:Real}},
								VVMR<:Vector{<:Vector{<:Matrix{<:Real}}},
								VVVR<:Vector{<:Vector{<:Vector{<:Real}}},
								VVVVR<:Vector{<:Vector{<:Vector{<:Vector{<:Real}}}},
								Tindex<:Indexθ}
	"transition matrix of the accumulator variable in the presence of input"
	Aᵃinput::VMR
	"scaling factor of the log-likelihood of choices"
	choiceLLscaling::R
	"a vector of the concatenated values of the parameters being fitted"
	concatenatedθ::VR
	"a structure indicating the index of each model parameter in the vector of concatenated values"
	indexθ::Tindex
	"indices of the parameters that influence the prior probabilities of the accumulator"
	indexθ_pa₁::VI=[1,4,8,10]
	"indices of the parameters that influence the transition probabilities of the accumulator"
	indexθ_paₜaₜ₋₁::VI=[1,2,3,5,7,9]
	"indices of the parameters that influence the lapse rate"
	indexθ_ψ::VI=[6]
	"posterior probabilities: element γ[s][j][t] corresponds to the p{a(t)=ξ(j),c(t)=k ∣ 𝐘} for the t-th time step in the s-th trialset"
	γ::VVVR
	"log-likelihood"
	ℓ::VR = fill(NaN,1)
	"maximum number of time steps across trials"
	maxtimesteps::TI
	"maximum number of clicks"
	maxclicks::TI
	"gradient of the log-likelihood with respect to glm parameters"
	∇ℓglm::VVθ
	"gradient of the log-likelihood with respect to all parameters, even those not being fit"
	∇ℓlatent::VR=zeros(length(fieldnames(FHMDDM.Latentθ)))
	"number of parameters that influence the prior probabilities of the accumulator"
	nθ_pa₁::TI = length(indexθ_pa₁)
	"number of parameters that influence the transition probabilities of the accumulator"
	nθ_paₜaₜ₋₁::TI = length(indexθ_paₜaₜ₋₁)
	"number of the parameters that influence the lapse rate"
	nθ_ψ::TI = length(indexθ_ψ)
	"Conditional likelihood of the emissions (spikes and/or choice) at each time bin. For time bins of each trial other than the last, it is the product of the conditional likelihood of all spike trains. For the last time bin, it corresponds to the product of the conditional likelihood of the spike trains and the choice. Element p𝐘𝑑[i][m][t][j,k] corresponds to ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k) across N neural units at the t-th time bin in the m-th trial of the i-th trialset. The last element p𝐘𝑑[i][m][end][j,k] of each trial corresponds to p(𝑑 | aₜ = ξⱼ, zₜ=k) ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k)"
	p𝐘𝑑::VVVVR
	"number of accumulator states"
	Ξ::TI
	"""
		dependent fields
	"""
	"transition matrix of the accumulator variable in the absence of input"
	Aᵃsilent::MR=copy(Aᵃinput[1])
	"normalization parameters in the forward-backward algorithm"
	D::VR = zeros(maxtimesteps)
	"forward terms"
	f::VVR = collect(zeros(Ξ) for t=1:maxtimesteps)
	"forward terms for the choice only chains"
	fᶜ::VVR = collect(zeros(Ξ) for t=1:maxtimesteps)
	"partial derivative of the initial probability of the accumulator"
	∇pa₁::VVR=collect(zeros(Ξ) for q=1:nθ_pa₁)
	"prior distribution of the accumulator"
	p𝐚₁::VR = zeros(eltype(Aᵃsilent),Ξ)
	"partial derivatives of the transition matrix of the accumulator variable in the presence of input"
	∇Aᵃinput::VVMR = collect(collect(zeros(Ξ,Ξ) for i = 1:nθ_paₜaₜ₋₁) for t=1:maxclicks)
	"partial derivatives of the transition matrix of the accumulator variable in the absence of input"
	∇Aᵃsilent::VMR = collect(zeros(Ξ,Ξ) for i = 1:nθ_paₜaₜ₋₁)
end

"""
	Memory and pre-computed quantities for obtaining the hessian of the log-likelihood of the choices
"""
@with_kw struct Memory_for_hessian_choiceLL{TI<:Integer,
											VI<:Vector{<:Integer},
											VR<:Vector{<:Real},
											MR<:Matrix{<:Real},
											VVR<:Vector{<:Vector{<:Real}},
											VMR<:Vector{<:Matrix{<:Real}},
											MVR<:Matrix{<:Vector{<:Real}},
											MMR<:Matrix{<:Matrix{<:Real}},
											VVVR<:Vector{<:Vector{<:Vector{<:Real}}},
											VVMR<:Vector{<:Vector{<:Matrix{<:Real}}},
											VMMR<:Vector{<:Matrix{<:Matrix{<:Real}}},
											PT<:Probabilityvector,
											VS<:Vector{<:Symbol}}
	"names of variables involved in specifying the hessian"
	parameternames::VS
	"number of accumulator states"
	Ξ::TI
	"total number of parameters used to compute the log-likelihood of choices"
	nθ::TI=length(parameternames)
	"indices of the parameters that influence the prior probabilities of the accumulator"
	indexθ_pa₁::VI
	"indices of the parameters that influence the transition probabilities of the accumulator"
	indexθ_paₜaₜ₋₁::VI
	"indices of the parameters that influence the lapse rate"
	indexθ_ψ::VI
	"number of parameters that influence the prior probabilities of the accumulator"
	nθ_pa₁::TI = length(indexθ_pa₁)
	"number of parameters that influence the transition probabilities of the accumulator"
	nθ_paₜaₜ₋₁::TI = length(indexθ_paₜaₜ₋₁)
	"number of the parameters that influence the lapse rate"
	nθ_ψ::TI = length(indexθ_ψ)
	"whether a parameter influences the prior probability of the accumulator, and if so, the index of that parameter"
	index_pa₁_in_θ::VI = let x = zeros(Int, nθ); x[indexθ_pa₁] .= 1:nθ_pa₁; x end
	"whether a parameter influences the transition probability of the accumulator, and if so, the index of that parameter"
	index_paₜaₜ₋₁_in_θ::VI = let x = zeros(Int, nθ); x[indexθ_paₜaₜ₋₁] .= 1:nθ_paₜaₜ₋₁; x end
	"whether a parameter influences the prior probability of the lapse, and if so, the index of that parameter"
	index_ψ_in_θ::VI = let x = zeros(Int, nθ); x[indexθ_ψ] .= 1:nθ_ψ; x end
	"`Probabilityvector`: a structure containing memory for computing the probability vector of the accumulator and the first- and second-order partial derivatives of the elements of the probability vector"
	P::PT
	"log-likelihood"
	ℓ::VR = zeros(1)
	"gradient of the log-likelihood"
	∇ℓ::VR=zeros(nθ)
	"hessian of the log-likelihood"
	∇∇ℓ::MR=zeros(nθ,nθ)
	"forward term. Element 'f[t][i]' corresponds to the t-th time step in a trial and i-th accumulator state"
	f::VVR
	"gradient of the forward term. Element '∇f[t][q][i]' corresponds to the t-th time step in a trial, q-th parameter, and i-th accumulator state"
	∇f::VVVR
	"gradient of the past-conditioned likelihood. Element `∇D[q]` corresponds to q-th parameter among all parameters"
	∇D::VR=zeros(nθ)
	"gradient of the backward term. Element '∇b[q][i]' corresponds to the q-th parameter and i-th accumulator state"
	∇b::VVR=collect(zeros(Ξ) for q=1:nθ)
	"transition matrix of the accumulator at a time step without auditory input. Element `Aᵃsilent[q][i,j]` corresponds to the transition probability p{a(t)=ξ(i) ∣ a(t-1) = ξ(j)}"
	Aᵃsilent::MR
	"first-order partial derivatives of the transition matrix of the accumulator at a time step without auditory input. Element `∇Aᵃsilent[q][i,j]` corresponds to the derivative of the transition probability p{a(t)=ξ(i) ∣ a(t-1) = ξ(j)} with respect to the q-th parameter that influence the accumulator transitions."
	∇Aᵃsilent::VMR
	"second-order partial derivatives of the transition matrix of the accumulator at a time step without auditory input. Element `∇∇Aᵃsilent[q,r][i,j]` corresponds to the derivative of the transition probability p{a(t)=ξ(i) ∣ a(t-1) = ξ(j)} with respect to the q-th parameter and r-th parameter that influence the accumulator transitions."
	∇∇Aᵃsilent::MMR
	"transition matrix of the accumulator at a time-step when there is input. Element `Aᵃinput[t][i,j]` corresponds to the t-th time step in a trial with input, i-th accumulator step in the current time step, and j-th accumulator state in the previous time step "
	Aᵃinput::VMR
	"partial derivatives of the transition matrix of the accumulator at a time-step when there is input. Element `∇Aᵃinput[q][t][i,j]` corresponds to the q-th drift-diffusion parameter, t-th time step in a trial with input, i-th accumulator step in the current time step, and j-th accumulator state in the previous time step "
	∇Aᵃinput::VVMR
	"second order partial derivatives of the transition matrix of the accumulator at a time-step when there is input. Element `∇∇Aᵃinput[q,r][t][i,j]` corresponds to the q-th and r-th drift-diffusion parameter, t-th time step in a trial with input, i-th accumulator step in the current time step, and j-th accumulator state in the previous time step "
	∇∇Aᵃinput::VMMR
	"first-order partial derivatives of the prior probability of the accumulator. Element `∇pa₁[q][i]` corresponds to the q-th parameter among the parameters that govern prior probability and i-th accumulator state"
	∇pa₁::VVR
	"second-order partial derivatives of the prior probability of the accumulator. Element `∇∇pa₁[q,r][i]` corresponds to the q-th and r-th parameter among the parameters that govern prior probability and i-th accumulator state"
	∇∇pa₁::MVR
	"derivative of the conditional likelihood of a choice with respect to the lapse parameter ψ. Element `∂p𝑑_∂ψ[i]` corresponds to the i-th accumulator state"
	∂p𝑑_∂ψ::VR=zeros(Ξ)
end

"""
	TrialSample

Behavioral choice and neuronal spike trains simulated by running the model forward in time using the auditory click
"""
@with_kw struct TrialSample{B<:Bool, VF<:Vector{<:AbstractFloat}, VVF<:Vector{<:Vector{<:AbstractFloat}}, VVI<:Vector{<:Vector{<:Integer}}}
	"a nested array whose element `accumulator[i][m][t]` is the value of the accumulated evidence on the t-th time step of the m-th trial on the i-th trialset"
	accumulator::VF
	"`true` mean a right choice and `false` a left choice"
	choice::B
	"a nested array whose element `λ[n][t]` is the value of generative spikes per s of the n-th neuron on the t-th time step of the m-th trial on the i-th trialset"
	λ::VVF
	"a nested array whose element `spiketrains[n][t]` is the spike train response of the n-th neuron on the t-th time step of the m-th trial on the i-th trialset"
	spiketrains::VVI
end

"""
	TrialsetSample

Simulated behavioral choice and neuronal spike trains on every trial of a trialset
"""
@with_kw struct TrialsetSample{VS<:Vector{<:TrialSample}}
	trials::VS
end

"""
	Sample

Behavioral choice and neuronal spike trains simulated by running the model forward in time using the auditory clicks on every trial of every trialset
"""
@with_kw struct Sample{VS<:Vector{<:TrialsetSample}}
	trialsets::VS
end

"""
	Expectation of the choice and spike train on a single trial

The unconditioned spike train of the n-th neuron on this trial can be computed as `spiketrain_leftchoice[n].*(1-rightchoice) .+ spiketrain_rightchoice[n].*rightchoice`

If no left choice occurred during simulations, then `spiketrain_leftchoice[n][t]` is zero for all `n` and `t`.

If we have a vector `𝐄` whose each element is a composite of the type `ExpectedEmissions` corresponding to a single trial, then the left-choice conditioned peri-stimulus time historam of the n-th neuron can be computed as follows
```
"""
@with_kw struct ExpectedEmissions{F<:AbstractFloat, VVF<:Vector{<:Vector{<:AbstractFloat}}}
	"expectation of the choice random variable. This can be interpreted as the expected probability of a right choice"
	rightchoice::F
	"expectation of the spike trains, conditioned on a left choice. The element `spiketrain_leftchoice[n][t]` corresponds to the n-th neuron at the t-th time step."
	spiketrain_leftchoice::VVF
	"expectation of the spike trains, conditioned on right choice"
	spiketrain_rightchoice::VVF
end


"""
	Quantities helpful for understanding the model
"""
@with_kw struct Characterization{VVE<:Vector{<:Vector{<:ExpectedEmissions}},
							VVF<:Vector{<:Vector{<:AbstractFloat}},
							VVVVF<:Vector{<:Vector{<:Vector{<:Vector{<:AbstractFloat}}}}}
	"probability of the accumulator variable given the actual auditory clicks. The element `paccumulator[i][m][t][j]` corresponds to the probability of the accumulator in the j-th state during the t-th time step of the m-th trial in the i-th trialset"
	paccumulator::VVVVF
	"posterior probability of the accumulator variable conditioned on the clicks and the behavioral choice. The format is identical to that of `paccumulator`."
	paccumulator_choice::VVVVF
	"posterior probability of the accumulator variable conditioned on the clicks, behavioral choice, and spike trains. The format is identical to that of `paccumulator`."
	paccumulator_choicespikes::VVVVF
	"posterior probability of the accumulator variable conditioned on the clicks and spike trains. The format is identical to that of `paccumulator`."
	paccumulator_spikes::VVVVF
	"log(base 2)-likelihood of the emissions on each trial. The element `LL[i][m]` is the log-likelihood of the choice in the m-th trial in the i-th trialset."
	LL::VVF
	"log(base 2)-likelihood of the observed behavioral choice. The element `LLchoice[i][m]` is the log-likelihood of the choice in the m-th trial in the i-th trialset."
	LLchoice::VVF
	"log(base 2)-likelihood of the observed choice under a Bernoulli model. The format is identical to that of the field `LLchoice`."
	LLchoice_bernoulli::VVF
	"log(base 2)-likelihood of the observed choice conditioned on the spike trains. The format is identical to that of the field `LLchoice`."
	LLchoice_spikes::VVF
	"log(base 2)-likelihood of the spike count at each time step. Element `LLspikes[i][n][m][t]` is the log-likelihood of observed spike count at the t-time step of the m-th trial for the n-th neuron in the i-th trialset."
	LLspikes::VVVVF
	"log(base 2)-likelihood of the spike count at each time step under a homogeneous Poisson model. The format is identical to that of the field `LLspikes`."
	LLspikes_poisson::VVVVF
	"expectation of the choice and spike trains computed averaging across simulations of the model. The element `expectedemissions[i][m]` contains the expected choice and the choice-conditioned spike trains of each neuron on the m-th trial of the i-th trialset."
	expectedemissions::VVE
end

"""
	ModelSummary

Features of the model useful for analysis
"""
@with_kw struct ModelSummary{F<:AbstractFloat,
							LT<:Latentθ,
							MF<:Matrix{<:AbstractFloat},
							VF<:Vector{<:AbstractFloat},
							VMF<:Vector{<:Matrix{<:AbstractFloat}},
							VVVF<:Vector{<:Vector{<:Vector{<:AbstractFloat}}},
							VVMF<:Vector{<:Vector{<:Matrix{<:AbstractFloat}}},
							VS<:Vector{<:String},
							VVGT<:Vector{<:Vector{<:GLMθ}},
							VVI<:Vector{<:Vector{<:Integer}}}
	"Element `designmatrix[i][n]` corresponds to the design matrix of the n-th neuron in the i-th trialset"
	designmatrix::VVMF
	"a coefficient used to scale up the glm parameters to mitigate ill-conditioning of the hessian"
	glm_parameter_scale_factor::F
	"Weighted inputs, except for those from the latent variables and the spike history, to each neuron on each time step in a trialset. The element `externalinputs[i][n][t]` corresponds to the input to the n-th neuron in the i-th trialset on the t-th time step in the trialset (time steps are across trials are concatenated)"
	externalinputs::VVVF
	"the log of the likelihood of the data given the parameters"
	loglikelihood::F
	"the log of the posterior probability of the parameters"
	logposterior::F
	"values of the parameters of the latent variable in their native space"
	thetanative::LT
	"values of the parameters of the latent variable mapped to real space"
	thetareal::LT
	"initial values of parameters of the latent variable in their space"
	theta0native::LT
	"parameters of each neuron's GLM. The element `θglm[i][n]` corresponds to the n-th neuron in the i-th trialset"
	thetaglm::VVGT
	"temporal basis vectors for the gain on each trial"
	temporal_basis_vectors_gain::VVMF
	"temporal basis vectors for the post-spike kernel"
	temporal_basis_vectors_postspike::VMF
	"temporal basis vectors for the post-stereoclick kernel"
	temporal_basis_vectors_poststereoclick::VMF
	"temporal basis vectors for the pre-movement kernel"
	temporal_basis_vectors_premovement::VMF
	"parameters concatenated into a vector"
	parametervalues::VF
	"name of each parameter"
	parameternames::VS
	"a vector of L2 penalty matrices"
	penaltymatrices::VMF
	"index of the parameters regularized by the L2 penalty matrices"
	penaltymatrixindices::VVI
	"cofficients of the penalty matrices"
	penaltycoefficients::VF
	"names of each L2 regularization penalty"
	penaltynames::VS
	"precision matrix of the gaussian prior on the parameters"
	precisionmatrix::MF
	"hessian of the log-likelihood function evaluated at the current parameters"
	hessian_loglikelihood::MF = fill(NaN, length(parametervalues), length(parametervalues))
	"hessian of the log-posterior function evaluated at the current parameters and hyperparameters"
	hessian_logposterior::MF = fill(NaN, length(parametervalues), length(parametervalues))
end

"""
	SpikeTrainLinearFilter

Linear filter used to smooth the spike train
"""
@with_kw struct SpikeTrainLinearFilter{VF<:Vector{<:AbstractFloat}}
	"the function with which the spike train is convolved"
	impulseresponse::VF
	"because of lack of spike train responses before the beginning of the trial, the initial response needs to reweighed"
	weights::VF
end

"""
	PerieventTimeHistogram

The mean across trials of a single condition (e.g. trials that ended with a left choice) of the filtered spike train of one neuron, and the estimated 95% confidence interval of the trial mean
"""
@with_kw struct PerieventTimeHistogram{CIM<:Bootstrap.ConfIntMethod,
									S<:String,
									STLF<:SpikeTrainLinearFilter,
									VF<:Vector{<:AbstractFloat}}
	"method used to compute the confidence interval. The default is the bias-corrected and accelerated confidence interval (Efron & Tibshirani, 1993) for a confidence level of 0.95. The confidence level is the fraction of time when a random confidence interval constructed using the method below contains the true peri-stimulus time histogram."
	confidence_interval_method::CIM = BCaConfInt(0.95)
	"the inear filter used to smooth the spike train"
	linearfilter::STLF
	"estimate of the lower limit of the confidence interval of the peri-event time histogram based on observed spike trains"
	lowerconfidencelimit::VF
	"estimate of the peri-event time histogram based on observed spike trains"
	observed::VF
	"estimate of the peri-event time histogram based on simulated spike trains"
	predicted::VF
	"An event in the trial (e.g. steroclick) corresponding to which the peri-stimulus time histogram is aligned, i.e., when time=0 is defined."
	referenceevent::S="stereoclick"
	"time, in seconds, from the reference event"
	time_s::VF
	"estimate of the upper limit of the confidence interval of the peri-event time histogram based on observed spike trains"
	upperconfidencelimit::VF
end

"""
	PETHSet

A set of peri-event time histogram of one neuron.
"""
@with_kw struct PETHSet{PETH<:PerieventTimeHistogram}
	"average across trials that ended in a left choice, including both correct and incorrect trials"
	leftchoice::PETH
	"average across trials on which the aggregate evidence favored left and the reward is baited on the left"
	leftevidence::PETH
	"average across trials on which the animal made a left choice and the evidence was strongly leftward. Therefore, only correct trials are included. Evidence strength is defined by the generative log-ratio of click rates: `γ` ≡ log(right click rate) - log(left click rate). Strong left evidence evidence include trials for which γ < -2.25"
	leftchoice_strong_leftevidence::PETH
	"average across trials on which the animal made a left choice and the generatative `γ` < -2.25 && `γ` < 0"
	leftchoice_weak_leftevidence::PETH
	rightchoice::PETH
	rightevidence::PETH
	rightchoice_strong_rightevidence::PETH
	rightchoice_weak_rightevidence::PETH
	"average across all trials"
	unconditioned::PETH
end

"""
	CVResults

Results of cross-validation
"""
@with_kw struct CVResults{C<:Characterization, VC<:Vector{<:CVIndices}, VS<:Vector{<:ModelSummary}, VVP<:Vector{<:Vector{<:PETHSet}}}
	"a composite containing quantities that are computed out-of-sample and used to characterize the model`"
	characterization::C
	"cvindices[k] indexes the trials and timesteps used for training and testing in the k-th resampling"
	cvindices::VC
	"post-stereoclick time histogram sets"
	psthsets::VVP
	"summaries of the training models"
	trainingsummaries::VS
end
