"""
	accumulatorprobability!(Aᵃinput, P, p𝐚, Aᵃsilent, θnative, trial)

Probability of the accumulator at each time step

MODIFIED ARGUMENT
-`Aᵃinput`: vector of matrices used as memory for computing the transition probability of the accumulator on time steps with stimulus input
-`p𝐚`: a vector whose element p𝐚[t][i] represents p(a[t] = ξ[i])

UNMODIFIED ARGUMENT
-`Aᵃsilent`: transition probability of the accumulator on timesteps without stimulus input
-`p𝐚₁`: prior distribution of the accumulator
-`trial`: structure containing information on a trial

"""
function accumulatorprobability!(p𝐚::Vector{<:Vector{<:AbstractFloat}},
								p𝐚₁::Vector{<:AbstractFloat},
 								Aᵃinput::Vector{<:Matrix{<:AbstractFloat}},
 								Aᵃsilent::Matrix{<:AbstractFloat},
								trial::Trial)
	p𝐚[1] .= p𝐚₁
	@inbounds for t=2:trial.ntimesteps
		if isempty(trial.clicks.inputindex[t])
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[trial.clicks.inputindex[t][1]]
		end
		p𝐚[t] = Aᵃ * p𝐚[t-1]
	end
	return nothing
end

"""
	accumulator_probability_given_choice!(p, choice, p𝐚_end, ψ)

Conditional distribution of the accumulator variable given the behavioral choice

MODIFIED ARGUMENT
-`p`: a vector serving as memory

UNMODIFIED ARGUMENT
-`Aᵃinput`: memory for computing the transition matrix during a timestep with stimulus input
-`Aᵃsilent`: transition matrix during a timestep without stimulus input
-`p𝐚`: distribution of the accumulator at the each time step of the trial
-`ψ`: lapse rate
-`trial`: a structure containing information on the trial being considered

"""
function accumulator_probability_given_choice!(p𝐚_𝑑::Vector{<:Vector{<:AbstractFloat}},
											p𝑑_𝐚::Vector{<:AbstractFloat},
											Aᵃinput::Vector{<:Matrix{<:AbstractFloat}},
											Aᵃsilent::Matrix{<:AbstractFloat},
											p𝐚::Vector{<:Vector{<:AbstractFloat}},
											ψ::AbstractFloat,
											trial::Trial)
	choicelikelihood!(p𝑑_𝐚, trial.choice, ψ) # `p𝐚_𝑑[ntimesteps]` now reprsents p(𝑑 ∣ a)
	p𝐚_𝑑[trial.ntimesteps] .= p𝑑_𝐚.*p𝐚[trial.ntimesteps] # `p𝐚_𝑑[ntimesteps]` now reprsents p(𝑑, a)
	D = sum(p𝐚_𝑑[trial.ntimesteps])
	p𝐚_𝑑[trial.ntimesteps] ./= D # `p𝐚_𝑑[ntimesteps]` now reprsents p(a ∣ 𝑑)
	b = ones(length(p𝑑_𝐚))
	for t = trial.ntimesteps-1:-1:1
		inputindex = trial.clicks.inputindex[t+1]
		if isempty(inputindex)
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[inputindex[1]]
		end
		if t+1 == trial.ntimesteps
			b = Aᵃ' * (p𝑑_𝐚.*b./D)
		else
			b = Aᵃ' * b
		end
		p𝐚_𝑑[t] = p𝐚[t] .* b
	end
	return nothing
end

"""
	collectpredictions(cvindices, 𝛌Δt)

Combine the predicted spike train response across cross-validation folds

ARGUMENT
-`cvindices`: indices of the trials and timesteps used for training and testing in each fold
-`𝛌Δt`: predicted spike trains, either conditioned on the choice or unconditioned. Element `𝛌Δt[f][i][n][τ]` corresponds to the f-the cross-validation fold, i-th trialset, n-th neuron, and τ-th time step among the time steps concatenated across the trials subsampled in the f-th cross-validation fold.

OUTPUT
-`λΔt`: predicted spike train response combined across cross-validation folds. Element `λΔt[i][n][τ]` corresponds to the i-th trialset, n-th neuron, and τ-th time step among the time steps concatenated across all trials in the i-th trialset.
"""
function collectpredictions(cvindices::Vector{<:CVIndices}, 𝛌Δt::Vector{<:Vector{<:Vector{<:Vector{<:AbstractFloat}}}})
	ntrialsets = length(cvindices[1].testingtrials)
	map(1:ntrialsets) do i
		ntimesteps = 0
		for f in eachindex(cvindices)
			ntimesteps += length(cvindices[f].testingtimesteps[i])
		end
		nneurons = length(𝛌Δt[1][i])
		map(1:nneurons) do n
			λΔt = fill(NaN, ntimesteps)
			for f in eachindex(cvindices)
				λΔt[cvindices[f].testingtimesteps[i]] .= 𝛌Δt[f][i][n]
			end
			return λΔt
		end
	end
end

"""
	collectpredictions(cvindices, 𝐏)

Combine the predicted distributions of a latent variable across cross-validation folds

ARGUMENT
-`cvindices`: indices of the trials and timesteps used for training and testing in each fold
-`𝐏`: predicted distribution of either the accumulator or the coupling variable, conditioned on both the spikes and the choices, conditioned on only the choices, or unconditioned. Element `𝐏[f][i][q][t][j]` corresponds to the probability of the latent variable being in the j-th state in the t-th time step of the q-th trial among the subsampled trials in the i-th trialset, evaluated in the f-th cross-validation fold.

RETURN
-`𝐩`: the predicted distribution. Element `𝐩[i][m][t][j]` corresponds to the probability of the latent variable being in the j-th state in the t-th time step of the m-th trial in the i-th trialset
"""
function collectpredictions(cvindices::Vector{<:CVIndices}, 𝐏::Vector{<:Vector{<:Vector{<:Vector{<:Vector{<:type}}}}}) where {type<:AbstractFloat}
	ntrialsets = length(cvindices[1].testingtrials)
	map(1:ntrialsets) do i
		ntrials = 0
		for cvindex in cvindices
			ntrials += length(cvindex.testingtrials[i])
		end
		𝐩 = collect([type[]] for m = 1:ntrials)
		for f in eachindex(cvindices)
			for q in eachindex(cvindices[f].testingtrials[i])
				m = cvindices[f].testingtrials[i][q]
				𝐩[m] = 𝐏[f][i][q]
			end
		end
		return 𝐩
	end
end

"""
	collectpredictions(cvindices, P𝑑)

Combine the predicted probabilities of behavioral choices across cross-validation folds

ARGUMENT
-`cvindices`: vector whose each element corresponds to indices of the trials and timesteps used for training and testing in each cross-validation fold
-`P𝑑`: predicted probabilities of behavioral choices. Element `P𝑑[f][i][q]` corresponds to the probability of the behavioral choice in the q-th trial among the subsampled trials in the i-th trialset, evaluated in the f-th cross-validation fold.
"""
function collectpredictions(cvindices::Vector{<:CVIndices}, P𝑑::Vector{<:Vector{<:Vector{<:AbstractFloat}}})
	ntrialsets = length(cvindices[1].testingtrials)
	map(1:ntrialsets) do i
		ntrials = 0
		for cvindex in cvindices
			ntrials += length(cvindex.testingtrials[i])
		end
		p𝑑 = fill(NaN, ntrials)
		for f in eachindex(cvindices)
			p𝑑[cvindices[f].testingtrials[i]] .= P𝑑[f][i]
		end
		return p𝑑
	end
end

"""
	Predictions(cvindices, testmodels)

Out-of-sample predictions

ARGUMENT
-`cvindices`: vector whose each element corresponds to indices of the trials and timesteps used for training and testing in each cross-validation fold
-`testmodels`: vector whose each element corresponds to a cross-validation fold. Each element contains a structure containing hold-out data and parameters learned using training data.

OUTPUT
-an instance of `Predictions`
"""
function Predictions(cvindices::Vector{<:CVIndices}, testmodels::Vector{<:Model})
	predictions_each_fold = collect(Predictions(testmodel) for testmodel in testmodels)
	collected_predictions = (FHMDDM.collectpredictions(cvindices, collect(getfield(predictions, field) for predictions in predictions_each_fold)) for field in (:p𝐚, :p𝐚_𝑑, :p𝐚_𝐘𝑑, :p𝐜_𝐘𝑑, :p𝑑, :λΔt, :λΔt_𝑑))
	return Predictions(collected_predictions..., predictions_each_fold[1].nsamples)
end

"""
	Predictions(model)

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of the model

RETURN
-a structure containing the predictions of the model
"""
function Predictions(model::Model; nsamples::Integer=100)
    @unpack trialsets, options, θnative = model
	@unpack Ξ, K = options
    λΔt = map(trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				zeros(trialset.ntimesteps)
			end
		  end
	λΔt_𝑑 = deepcopy(λΔt)
	p𝐚 = map(trialsets) do trialset
			map(trialset.trials) do trial
				collect(zeros(Ξ) for t=1:trial.ntimesteps)
			end
		  end
	p𝐜_𝐘𝑑 = map(trialsets) do trialset
			map(trialset.trials) do trial
				collect(zeros(K) for t=1:trial.ntimesteps)
			end
		  end
	p𝐚_𝑑, p𝐚_𝐘𝑑 = deepcopy(p𝐚), deepcopy(p𝐚)
	p𝑑 = collect(zeros(trialset.ntrials) for trialset in trialsets)
	memory = Memoryforgradient(model)
	P = FHMDDM.update!(memory, model, concatenateparameters(model)[1])
	@unpack Aᵃinput, Aᵃsilent, Aᶜ, p𝐚₁, πᶜ = memory
	f⨀b = memory.f
	p𝑑_𝐚 = ones(Ξ)
	maxtimesteps = length(f⨀b)
	a = zeros(Int, maxtimesteps)
	c = zeros(Int, maxtimesteps)
	𝐄𝐞_𝐡_𝛚 = map(trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				return externalinput(mpGLM), postspikefilter(mpGLM), transformaccumulator(mpGLM)
			end
		end
    for trialset in trialsets
		for trial in trialset.trials
			i = trial.trialsetindex
			m = trial.index_in_trialset
			𝛕 = trial.τ₀ .+ (1:trial.ntimesteps)
			forward!(memory, P, θnative, trial)
			backward!(memory, P, trial)
			accumulatorprobability!(p𝐚[i][m], p𝐚₁, Aᵃinput, Aᵃsilent, trial)
			accumulator_probability_given_choice!(p𝐚_𝑑[i][m], p𝑑_𝐚, Aᵃinput, Aᵃsilent, p𝐚[i][m], θnative.ψ[1], trial)
			for t = 1:trial.ntimesteps
				p𝐚_𝐘𝑑[i][m][t] = dropdims(sum(f⨀b[t], dims=2), dims=2)
				p𝐜_𝐘𝑑[i][m][t] = dropdims(sum(f⨀b[t], dims=1), dims=1)
			end
			for s = 1:nsamples
				samplecoupling!(c, Aᶜ, trial.ntimesteps, πᶜ)
				sampleaccumulator!(a, Aᵃinput, Aᵃsilent, p𝐚₁, trial)
				p𝑑[i][m] += sample(a[trial.ntimesteps], θnative.ψ[1], Ξ)/nsamples
				for (𝐄𝐞_𝐡_𝛚, λΔt, mpGLM) in zip(𝐄𝐞_𝐡_𝛚[i], λΔt[i], trialset.mpGLMs)
					λΔt[𝛕] .+= sample(a, c, 𝐄𝐞_𝐡_𝛚[1], 𝐄𝐞_𝐡_𝛚[2], mpGLM, 𝐄𝐞_𝐡_𝛚[3], 𝛕)./nsamples
				end
				sample_accumulator_given_choice!(a, Aᵃinput, Aᵃsilent, p𝐚[i][m], p𝐚_𝑑[i][m][trial.ntimesteps], trial)
				 for (𝐄𝐞_𝐡_𝛚, λΔt_𝑑, mpGLM) in zip(𝐄𝐞_𝐡_𝛚[i], λΔt_𝑑[i], trialset.mpGLMs)
					λΔt_𝑑[𝛕] .+= sample(a, c, 𝐄𝐞_𝐡_𝛚[1], 𝐄𝐞_𝐡_𝛚[2], mpGLM, 𝐄𝐞_𝐡_𝛚[3], 𝛕)./nsamples
				end
			end
		end
	end
    return Predictions(	p𝐚 = p𝐚,
						p𝐚_𝑑 = p𝐚_𝑑,
						p𝐚_𝐘𝑑 = p𝐚_𝐘𝑑,
						p𝐜_𝐘𝑑 = p𝐜_𝐘𝑑,
						p𝑑 = p𝑑,
						λΔt = λΔt,
						λΔt_𝑑 = λΔt_𝑑,
						nsamples = nsamples)
end

"""
	sampleaccumulator!(a, Aᵃinput, Aᵃsilent, p𝐚₁, trial)

Sample the values of the accumulator variable in one trial

MODIFIED ARGUMENT
-`a`: a vector containing the sample value of the coupling variable in each time step

UNMODIFIED ARGUMENT
-`Aᵃinput`: memory for computing the transition matrix during a timestep with stimulus input
-`Aᵃsilent`: transition matrix during a timestep without stimulus input
-`p𝐚₁`: prior distribution of the accumulator
-`trial`: a structure containing information on the trial being considered
"""
function sampleaccumulator!(a::Vector{<:Integer}, Aᵃinput::Vector{<:Matrix{<:Real}}, Aᵃsilent::Matrix{<:Real}, p𝐚₁::Vector{<:AbstractFloat}, trial::Trial)
	a[1] = findfirst(rand() .< cumsum(p𝐚₁))
	for t = 2:trial.ntimesteps
		if isempty(trial.clicks.inputindex[t])
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[trial.clicks.inputindex[t][1]]
		end
		p𝐚ₜ_aₜ₋₁ = Aᵃ[:,a[t-1]]
		a[t] = findfirst(rand() .< cumsum(p𝐚ₜ_aₜ₋₁))
	end
	return nothing
end

"""
	sample_accumulator_given_choice!(a, Aᵃinput, Aᵃsilent, p𝐚_𝑑, trial)

A sample of the accumulator in one trial conditioned on the behavioral choice

MODIFIED ARGUMENT
-`a`: a vector representing the value of the accumulator at each time step of the trial

UNMODIFIED ARGUMENT
-`Aᵃinput`: vector of matrices used as memory for computing the transition probability of the accumulator on time steps with stimulus input
-`Aᵃsilent`: transition probability of the accumulator on timesteps without stimulus input
-`p𝐚`: probability of the accumulator at each time step of the trial
-`p𝐚_end_𝑑`: posterior probability of the accumulator, given the choice, at the last time step. The i-th element represents p(a=ξᵢ ∣ 𝑑)
-`trial`: structure containing information on a trial
"""
function sample_accumulator_given_choice!(a::Vector{<:Integer},
										Aᵃinput::Vector{<:Matrix{<:AbstractFloat}},
 										Aᵃsilent::Matrix{<:AbstractFloat},
										p𝐚::Vector{<:Vector{<:AbstractFloat}},
										p𝐚_end_𝑑::Vector{<:AbstractFloat},
										trial::Trial)
	a[trial.ntimesteps] = findfirst(rand() .< cumsum(p𝐚_end_𝑑))
	for t = trial.ntimesteps-1:-1:1
		inputindex = trial.clicks.inputindex[t+1]
		if isempty(inputindex)
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[inputindex[1]]
		end
		p_𝐚ₜ_aₜ₊₁ = Aᵃ[a[t+1],:] .* p𝐚[t] ./ p𝐚[t+1][a[t+1]]
		a[t] = findfirst(rand() .< cumsum(p_𝐚ₜ_aₜ₊₁))
	end
	return nothing
end

"""
	samplecoupling!(c, Aᶜ, ntimesteps, πᶜ)

Sample the values of the coupling variable in one trial

MODIFIED ARGUMENT
-`c`: a vector containing the sample value of the coupling variable in each time step

ARGUMENT
-`Aᶜ`: transition matrix of the coupling variable
-`ntimesteps`: number of time steps in the trial
-`πᶜ`: prior probability of the coupling variable
"""
function samplecoupling!(c::Vector{<:Integer}, Aᶜ::Matrix{<:Real}, ntimesteps::Integer, πᶜ::Vector{<:Real})
	if length(πᶜ) == 1
		c .= 1
	else
		cumulativep𝐜 = cumsum(πᶜ)
	    c[1] = findfirst(rand() .< cumulativep𝐜)
		cumulativeAᶜ = cumsum(Aᶜ, dims=1)
	    for t = 2:ntimesteps
	        cumulativep𝐜 = cumulativeAᶜ[:,c[t-1]]
	        c[t] = findfirst(rand() .< cumulativep𝐜)
	    end
	end
	return nothing
end

"""
	sample(a_end, ψ, Ξ)

Sample a choice on a trial

ARGUMENT
-`a_end`: state of the accumulator at the last time step of the trial
-`ψ`: lapse rate
-`Ξ`: number of states that the accumulator can take
"""
function sample(a_end::Integer, ψ::AbstractFloat, Ξ::Integer)
	zeroindex = cld(Ξ,2)
	if a_end < zeroindex
		p_right_choice = ψ/2
	elseif a_end > zeroindex
		p_right_choice = 1-ψ/2
	else a_end == zeroindex
		p_right_choice = 0.5
	end
	choice = rand() < p_right_choice
end

"""
	sample(a, c, 𝛕, mpGLM)

Generate a sample of spiking response on each time step of one trial

ARGUMENT
-`a`: a vector representing the state of the accumulator at each time step of a trial. Note that length(a) >= length(𝛕).
-`c`: a vector representing the state of the coupling variable at each time step. Note that length(c) >= length(𝛕).
-`𝐄𝐞`: input from external events
-`𝐡`: value of the post-spikefilter at each time lag
-`mpGLM`: a structure containing information on the mixture of Poisson GLM of a neuron
-`𝛚`: transformed values of the accumulator
-`𝛕`: time steps in the trialset. The number of time steps in the trial corresponds to the length of 𝛕.

RETURN
-`𝐲̂`: a vector representing the sampled spiking response at each time step
"""
function sample(a::Vector{<:Integer}, c::Vector{<:Integer}, 𝐄𝐞::Vector{<:AbstractFloat}, 𝐡::Vector{<:AbstractFloat}, mpGLM::MixturePoissonGLM, 𝛚::Vector{<:AbstractFloat}, 𝛕::UnitRange{<:Integer})
	@unpack Δt, 𝐕, 𝐲, Ξ = mpGLM
	@unpack 𝐠, 𝐮, 𝐯, 𝛃, fit_𝛃 = mpGLM.θ
	max_spikehistory_lag = length(𝐡)
	K𝐠 = length(𝐠)
	K𝐯 = length(𝐯)
	max_spikes_per_step = floor(1000Δt)
    𝐲̂ = zeros(Int, length(𝛕))
    for t = 1:length(𝛕)
        τ = 𝛕[t]
        j = a[t]
        k = c[t]
		gₖ = 𝐠[min(k, K𝐠)]
		if fit_𝛃 && (j==1 || j==Ξ)
			𝐰ₖ = 𝛃[min(k, K𝐯)]
		else
			𝐰ₖ = 𝐯[min(k, K𝐯)]
		end
		L = gₖ + 𝐄𝐞[τ]
		for i in eachindex(𝐰ₖ)
			L+= 𝛚[j]*𝐕[τ,i]*𝐰ₖ[i]
		end
		for lag = 1:min(max_spikehistory_lag, t-1)
			if 𝐲̂[t-lag] > 0
				L += 𝐡[lag]*𝐲̂[t-lag]
			end
		end
        λ = softplus(L)
        𝐲̂[t] = min(rand(Poisson(λ*Δt)), max_spikes_per_step)
    end
	return 𝐲̂
end

"""
    sample(model)

Generate latent and emission variables for all trials of all trialsets

ARGUMENT
-`model`: an instance of the factorial-hidden Markov drift-diffusion model

OUTPUT
-a structure with data sampled from the parameters of the model
"""
function sample(model::Model; folderpath::String = dirname(model.options.datapath))
	predictions = Predictions(model; nsamples=1)
	newtrialsets = 	map(model.trialsets, predictions.p𝑑, predictions.λΔt) do trialset, p𝑑, λΔt
						newtrials =	map(trialset.trials, p𝑑) do oldtrial, p𝑑
										Trial(clicks=oldtrial.clicks,
						                      choice=Bool(p𝑑),
											  movementtime_s=oldtrial.movementtime_s,
						                      ntimesteps=oldtrial.ntimesteps,
						                      previousanswer=oldtrial.previousanswer,
											  index_in_trialset=oldtrial.index_in_trialset,
											  τ₀=oldtrial.τ₀,
											  trialsetindex=oldtrial.trialsetindex)
									end
						new_mpGLMs = map(trialset.mpGLMs, λΔt) do old_mpGLM, λΔt
										values = map(fieldnames(MixturePoissonGLM)) do fieldname
													if fieldname == :θ
														FHMDDM.copy(old_mpGLM.θ)
													elseif fieldname == :𝐲
														convert.(Int,λΔt)
													else
														getfield(old_mpGLM, fieldname)
													end
												end
										MixturePoissonGLM(values...)
									end
						Trialset(trials=newtrials, mpGLMs=new_mpGLMs)
					end
		options = dictionary(model.options)
		options["datapath"] = joinpath(folderpath,"data.mat")
		options["resultspath"] = joinpath(folderpath,"results.mat")
		Model(options=Options(model.options.nunits, options),
			gaussianprior=GaussianPrior(model.options, newtrialsets),
			θnative=FHMDDM.copy(model.θnative),
			θreal=FHMDDM.copy(model.θreal),
			θ₀native=FHMDDM.copy(model.θ₀native),
			trialsets=newtrialsets)
end

"""
	samples(model, nsamples)

Generate and save samples of the data

ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model
-`nsamples`: number of samples to make
"""
function samples(model::Model, nsamples::Integer)
	@assert nsamples > 0
	pad = ceil(Int, log10(nsamples))
	open(joinpath(dirname(model.options.datapath), "samplepaths.txt"), "w") do io
	    for i=1:nsamples
	        folderpath = joinpath(dirname(model.options.datapath),"sample"*string(i;pad=pad))
	        !isdir(folderpath) && mkdir(folderpath)
	        filepath = joinpath(folderpath, "data.mat")
	        println(io, filepath)
	        sampledmodel = sample(model; folderpath=folderpath)
	        savedata(sampledmodel)
	    end
	end
	return nothing
end

"""
	sampleclicks(a_latency_s, clickrate_Hz, Δt, ntimesteps, right2left)

Create a structure containing information on a sequence of simulated clicks

INPUT
-`a_latency_s`: latency, in second, of the response of the accumulator to the clicks
-`clickrate_Hz`: number of left and right clicks, combined, per second
-`Δt`: size of the time step
-`ntimesteps`: number of time steps in the trial
-`right2left`: ratio of the right to left click rate

OPTIONAL INPUT
-`rng`: random number generator

RETURN
-a structure containing the times and time step indices of simulated clicks

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> FHMDDM.sampleclicks(0.01, 40, 0.01, 100, 30; rng=MersenneTwister(1234))
	Clicks{Vector{Float64}, BitVector, Vector{Int64}, Vector{Vector{Int64}}}
	  time: Array{Float64}((46,)) [0.0479481935162798, 0.06307130174886962, 0.0804820073564533, 0.11317136052678396, 0.18273464895638575, 0.2000809403010865, 0.2086987723064543, 0.21917011781456938, 0.23527419909502842, 0.25225718711259393  …  0.8251247971779945, 0.8461572549605891, 0.847170493491451, 0.8519321105940183, 0.8555972472927873, 0.8670437145405672, 0.93879550239758, 0.9419273975453288, 0.9484616835697396, 0.9755875605263443]
	  inputtimesteps: Array{Int64}((35,)) [6, 8, 10, 13, 20, 22, 23, 25, 27, 30  …  77, 79, 83, 84, 86, 87, 88, 95, 96, 99]
	  inputindex: Array{Vector{Int64}}((100,))
	  source: BitVector
	  left: Array{Vector{Int64}}((100,))
	  right: Array{Vector{Int64}}((100,))
```
"""
function sampleclicks(a_latency_s::Real,
					  clickrate_Hz::Real,
					  Δt::Real,
					  ntimesteps::Integer,
					  right2left::Real;
					  rng::AbstractRNG=MersenneTwister())
	leftrate = clickrate_Hz/(1+right2left)
	rightrate = clickrate_Hz - leftrate
	duration_s = ntimesteps*Δt
	leftclicktimes = samplePoissonprocess(leftrate, duration_s; rng=rng)
	rightclicktimes = samplePoissonprocess(rightrate, duration_s; rng=rng)
	Clicks(a_latency_s, Δt, leftclicktimes, ntimesteps, rightclicktimes)
end

"""
	samplePoissonprocess(λ, T)

Return the event times from sampling a Poisson process with rate `λ` for duration `T`

INPUT
-`λ`: expected number of events per unit time
-`T`: duration in time to simulate the process

OPTIONAL INPUT
-`rng`: random number generator

RETURN
-a vector of event times

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> FHMDDM.samplePoissonprocess(10, 1.0; rng=MersenneTwister(1234))
	6-element Vector{Float64}:
	 0.24835053723904896
	 0.40002089777669625
	 0.4604645464869504
	 0.5300512053508091
	 0.6607031685057758
	 0.9387319245195712
```
"""
function samplePoissonprocess(λ::Real,
							  T::Real;
							  rng::AbstractRNG=MersenneTwister())
	@assert λ > 0
	@assert T > 0
	times = zeros(1)
	while times[end] < T
		times = vcat(times, times[end]+randexp(rng)/λ)
	end
	return times[2:end-1]
end
