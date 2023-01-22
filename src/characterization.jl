"""
	Characterization(model)

Compute quantities useful for characterizing the model

ARGUMENT
-a composite containing the data, parameters, and hyperparameter of a factoral hidden Markov drift-diffusion model

OPTIONAL ARGUMENT
-`nsamples`: number of samples to include within the composite

RETURN
-a composite containg quantities useful for understanding the model. See `types.jl` for details on each field of the composite.
"""
Characterization(model::Model; nsamples=100) = Characterization(model, model;nsamples=nsamples)

"""
	Characterization(testmodel, trainingmodel)
"""
function Characterization(testmodel::Model, trainingmodel::Model; nsamples::Integer=100)
	Characterization(paccumulator = accumulator_distribution(testmodel),
					paccumulator_choice = posterior_accumulator_distribution(testmodel; conditionedon="choices"),
					paccumulator_choicespikes = posterior_accumulator_distribution(testmodel; conditionedon="choices_spikes"),
					paccumulator_spikes = posterior_accumulator_distribution(testmodel; conditionedon="spikes"),
					LLchoice = loglikelihood_choice(testmodel),
					LLchoice_bernoulli = map((testset, trainingset)->loglikelihood_choice_bernoulli(testset, trainingset), testmodel.trialsets, trainingmodel.trialsets),
					LLchoice_spikes = loglikelihood_choice_given_spikes(testmodel),
					LLspikes = loglikelihood_spiketrains(testmodel),
					LLspikes_poisson = map((testset, trainingset)->loglikelihood_spiketrains_poisson(testset, trainingset), testmodel.trialsets, trainingmodel.trialsets),
					expectedemissions = ExpectedEmissions(testmodel, nsamples))
end

"""
	accumulator_distribution(model)

Moment-to-moment probability distribution of the accumulated evidence

ARGUMENT
-a composite containing the data, parameters, and hyperparameter of a factoral hidden Markov drift-diffusion model

RETURN
-`p𝐚`: a nested array whose element `p𝐚[i][m][t][j]` corresponds to the probability of the accumulator in the j-th state during the t-th time step of the m-th trial in the i-th trialset
"""
function accumulator_distribution(model::Model)
	memory = Memoryforgradient(model)
	P = update_for_latent_dynamics!(memory, model.options, model.θnative)
	map(model.trialsets) do trialset
		map(trialset.trials) do trial
			p𝐚 = collect(zeros(0) for t=1:trial.ntimesteps)
			accumulator_prior_transitions!(memory.Aᵃinput, P, memory.p𝐚₁, trial)
			p𝐚[1] = copy(memory.p𝐚₁)
			@inbounds for t=2:trial.ntimesteps
				Aᵃ = transitionmatrix(trial.clicks, memory.Aᵃinput, memory.Aᵃsilent, t)
				p𝐚[t] = Aᵃ * p𝐚[t-1]
			end
			return p𝐚
		end
	end
end

"""
	posterior_accumulator_distribution

Moment-to-moment distribution of the accumulated evidence given the choices, spike trains, or both

ARGUMENT
-`model`:a composite containing the data, parameters, and hyperparameter of a factoral hidden Markov drift-diffusion model

OPTIONAL ARGUMENT
-`conditionedon`: a string that needs to contain "choice" or "spike"

RETURN
-`𝐩`: a nested array whose element `𝐩[i][m][t][j]` corresponds to the probability of the accumulator in the j-th state during the t-th time step of the m-th trial in the i-th trialset

"""
function posterior_accumulator_distribution(model::Model; conditionedon::String="choices_spikes")
	memory = Memoryforgradient(model)
	if occursin("choice", conditionedon) && occursin("spike", conditionedon)
		posteriors!(memory, model)
	elseif occursin("choice", conditionedon)
		choiceposteriors!(memory, model)
	elseif occursin("spike", conditionedon)
		posterior_on_spikes!(memory, model)
	else
		error("unrecognized data type")
	end
	fb = sortbytrial(memory.γ, model)
	map(fb) do fb
		map(fb) do fb
			map(fb) do fb
				dropdims(sum(fb, dims=2), dims=2)
			end
		end
	end
end

"""
	loglikelihood_choice(model)

Log(base 2)-likelihood of the choice on each trial

ARGUMENT
-a composite containing the data, parameters, and hyperparameter of a factoral hidden Markov drift-diffusion model

RETURN
-`ℓ₂𝑑`: a nested array whose element `ℓ₂𝑑[i][m]` is the log-likelihood of the choice in the m-th trial in the i-th trialset.
"""
function loglikelihood_choice(model::Model)
	memory = Memoryforgradient(model)
	P = update_for_∇latent_dynamics!(memory, model.options, model.θnative)
	map(model.trialsets) do trialset
		map(trialset.trials) do trial
			accumulator_prior_transitions!(memory.Aᵃinput, P, memory.p𝐚₁, trial)
			p𝐚 = memory.p𝐚₁
			for t = 2:trial.ntimesteps
				Aᵃ = transitionmatrix(trial.clicks, memory.Aᵃinput, memory.Aᵃsilent, t)
				p𝐚 = Aᵃ*p𝐚
			end
			p𝑑_𝐚 = memory.p𝑑_a[trial.trialsetindex][trial.index_in_trialset]
			conditionallikelihood!(p𝑑_𝐚, trial.choice, model.θnative.ψ[1])
			log2(dot(p𝐚, p𝑑_𝐚))
		end
	end
end

"""
	loglikelihood_choice_given_spikes(model)

Log(base 2)-likelihood of the choice on each trial, conditioned on the spike trains

ARGUMENT
-`model`:a composite containing the data, parameters, and hyperparameter of a factoral hidden Markov drift-diffusion model

RETURN
-`ℓ₂𝑑_𝐘`: a nested array whose element ℓ₂𝑑_𝐘[i][m]` is the log(base 2)-likelihood of the choice in the m-th trial in the i-th trialset.
"""
function loglikelihood_choice_given_spikes(model::Model)
	memory = Memoryforgradient(model)
	P = update!(memory, model)
	memory_𝐘 = Memoryforgradient(model)
	for i in eachindex(memory_𝐘.p𝐘𝑑)
		scaledlikelihood!(memory_𝐘.p𝐘𝑑[i], model.options.sf_y, model.trialsets[i])
	end
	P_𝐘 = update_for_latent_dynamics!(memory_𝐘, model.options, model.θnative)
	log2e = log2(exp(1))
	map(model.trialsets) do trialset
		map(trialset.trials) do trial
			memory.ℓ[1] = 0.0
			memory_𝐘.ℓ[1] = 0.0
			forward!(memory, P, model.θnative, trial)
			forward!(memory_𝐘, P_𝐘, model.θnative, trial)
			return log2e*(memory.ℓ[1] - memory_𝐘.ℓ[1])
		end
	end
end

"""
	loglikelihood_choice_bernoulli(test_trialset, training_trialset)

Log-likelihood of the choices under a Bernoulli model

ARGUMENT
-`test_trialset`: held-out trials
-`training_trialset`: trials used to estimate the parameter of the Bernoulli distribution

RETURN
-`ℓ₂𝑑_bernoulli`: nested array whose element `ℓ₂𝑑_bernoulli[i][m]` is the log-likelihood of the choice in the m-th trial in the i-th trialset.
"""
function loglikelihood_choice_bernoulli(test_trialset::Trialset, training_trialset::Trialset)
	p = 0.0
	for trainingtrial in training_trialset.trials
		p+= trainingtrial.choice
	end
	p/=length(training_trialset.trials)
	log₂p = log2(p)
	log₂q = log2(1-p)
	map(test_trialset.trials) do testtrial
		testtrial.choice ? log₂p : log₂q
	end
end

"""
	loglikelihood_spiketrains_poisson(test_trialset, training_trialset)

Log-likelihood of the spike trains under a Bernoulli model

ARGUMENT
-`test_trialset`: held-out trials
-`training_trialset`: trials used to estimate the parameter of the Bernoulli distribution

RETURN
-`ℓ₂𝐲_poisson`: a nested array whose element `ℓ₂𝐲_poisson[m][n][t]` is the log-likelihood of observed spike count at the t-time step of the m-th trial for the n-th neuron.
"""
function loglikelihood_spiketrains_poisson(test_trialset::Trialset, training_trialset::Trialset)
	concatenatedℓs = map(test_trialset.mpGLMs, training_trialset.mpGLMs) do test_mpGLM, training_mpGLM
						λΔt = mean(training_mpGLM.𝐲)
						map(y->log2(poissonlikelihood(λΔt, y)), test_mpGLM.𝐲)
					end
	map(test_trialset.trials) do trial
		timesteps = trial.τ₀ .+ (1:trial.ntimesteps)
		map(concatenatedℓs) do concatenatedℓ
			concatenatedℓ[timesteps]
		end
	end
end

"""
	loglikelihood_spiketrains(model)

Log(base 2)-likelihood of the spike count of each neuron at each time step

ARGUMENT
-`model`:a composite containing the data, parameters, and hyperparameter of a factoral hidden Markov drift-diffusion model

RETURN
-`ℓ₂𝐲`: a nested array whose element `ℓ₂𝐲[i][m][n][t]` is the log-likelihood of the spike count of n-th neuron of the i-th trialset on the t-th time step of the m-th trial within the i-th trialset.
"""
function loglikelihood_spiketrains(model::Model)
	memory = Memoryforgradient(model)
	P = update_for_∇latent_dynamics!(memory, model.options, model.θnative)
	p𝑦_𝐚𝐜 = zeros(model.options.Ξ, model.options.K)
	map(model.trialsets) do trialset
		map(trialset.trials) do trial
			ℓ₂𝐲 = collect(zeros(trial.ntimesteps) for mpGLM in trialset.mpGLMs)
			accumulator_prior_transitions!(memory.Aᵃinput, P, memory.p𝐚₁, trial)
			p𝐚 = memory.p𝐚₁
			p𝐜 = memory.πᶜ
			for t=1:trial.ntimesteps
				if t > 1
					Aᵃ = transitionmatrix(trial.clicks, memory.Aᵃinput, memory.Aᵃsilent, t)
					p𝐚 = Aᵃ*p𝐚
					p𝐜 = memory.Aᶜ*p𝐜
				end
				τ = trial.τ₀ + t
				for (mpGLM, ℓ₂𝐲) in zip(trialset.mpGLMs, ℓ₂𝐲)
					conditionallikelihood!(p𝑦_𝐚𝐜, mpGLM, τ)
					ℓ₂𝐲[t] = log2(p𝐚'*p𝑦_𝐚𝐜*p𝐜)
				end
			end
			return ℓ₂𝐲
		end
	end
end

"""
	ExpectedEmissions(model, nsamples)

Compute the expectation of the choice and spike train on each trial by averaging across simulations

While the expectation of the choice can be computed without simulation, the expectation of a spike train requires simulation due to the spike history input.

Note the similarity between this function and `drawsamples`

ARGUMENT
-`model`: a composite containing the data, parameters, and hyperparameters of the factorial hidden-Markov drift-diffusion model
-`nsamples`: number of samples to draw

RETURN
-`expectedemissions`: a nested array whose element `expectedemissions[i][m]` contains the expected choice and the choice-conditioned spike trains of each neuron on the m-th trial of the i-th trialset.
"""
function ExpectedEmissions(model::Model, nsamples)
	memory = Memoryforgradient(model)
	P = update_for_latent_dynamics!(memory, model.options, model.θnative)
	a = zeros(Int, memory.maxtimesteps)
	c = zeros(Int, memory.maxtimesteps)
	map(model.trialsets) do trialset
		𝐄𝐞 = map(mpGLM->externalinput(mpGLM), trialset.mpGLMs)
		𝐡 = map(mpGLM->postspikefilter(mpGLM), trialset.mpGLMs)
		𝛚 = map(mpGLM->transformaccumulator(mpGLM), trialset.mpGLMs)
		map(trialset.trials) do trial
			accumulator_prior_transitions!(memory.Aᵃinput, P, memory.p𝐚₁, trial)
			E𝐘left = collect(zeros(trial.ntimesteps) for mpGLM in trialset.mpGLMs)
			E𝐘right = deepcopy(E𝐘left)
			nright = 0
			for s = 1:nsamples
				trialsample = sampletrial!(a, c, 𝐄𝐞, 𝐡, memory, 𝛚, model.θnative.ψ[1], trial, trialset)
				nright += trialsample.choice
				E𝐘 = trialsample.choice ? E𝐘right : E𝐘left
				for (E𝐲, 𝐲) in zip(E𝐘, trialsample.spiketrains)
					for t in eachindex(E𝐲)
						E𝐲[t] += 𝐲[t]
					end
				end
			end
			nleft = nsamples-nright
			if nleft > 0
				for E𝐲left in E𝐘left
					for t in eachindex(E𝐲left)
						E𝐲left[t] /= nleft
					end
				end
			end
			if nright > 0
				for E𝐲right in E𝐘right
					for t in eachindex(E𝐲right)
						E𝐲right[t] /= nright
					end
				end
			end
			ExpectedEmissions(rightchoice=nright/nsamples,
							spiketrain_leftchoice=E𝐘left,
							spiketrain_rightchoice=E𝐘right)
		end
	end
end

"""
	Characterization(cvindices, testmodels, trainingmodels)

Out-of-sample computation of quantities for characterizing the model

ARGUMENT
-`cvindices`: a vector of composites, each of which containing the indices of trials used for testing and training for a particular cross-validation fold
-`testmodels`: a vector of composites, each of which containing the held-out data for a cross-validation fold
-`trainingmodels`: a vector of composites, each of which containing the training data for a cross-validation fold
"""
function Characterization(cvindices::Vector{<:CVIndices}, testmodels::Vector{<:Model}, trainingmodels::Vector{<:Model}; nsamples::Integer=100)
	characterization_each_fold = map((testmodel, trainingmodel)->Characterization(testmodel, trainingmodel; nsamples=nsamples), testmodels, trainingmodels)
	ntrialsets = length(cvindices[1].testingtrials)
	trialindices = collect(vcat((cvindex.testingtrials[i] for cvindex in cvindices)...) for i=1:ntrialsets)
	values =
		map(fieldnames(Characterization)) do fieldname
			map(1:ntrialsets) do i
				field = vcat((getfield(characterization, fieldname)[i] for characterization in characterization_each_fold)...)
				field[trialindices[i]]
			end
		end
	Characterization(values...)
end

"""
	save(characterization, folderpath)

Save the characterizations of the model.

Each field of the composite `characterization` is saved within a separate file, with the same name as that of the field, within a folder whose absolute path is specified by `folderpath`.
"""
function save(characterization::Characterization, folderpath::String)
	if !isdir(folderpath)
		mkdir(folderpath)
		@assert isdir(folderpath)
	end
	for fieldname in fieldnames(Characterization)
		filepath = joinpath(folderpath, String(fieldname)*".mat")
		if fieldname==:expectedemissions
			entry =
				map(characterization.expectedemissions) do expectedemissions
					map(expectedemissions) do expectedemissions
						dictionary(expectedemissions)
					end
				end
		else
			entry = getfield(characterization, fieldname)
		end
		dict = Dict(String(fieldname)=>entry)
	    matwrite(filepath, dict)
	end
end

"""
"""
function peristimulus_time_histogram(𝐄::Vector{<:ExpectedEmissions}, options::Options)
	maxtimesteps = maximum((length(E.spiketrain_leftchoice[1]) for E in 𝐄))
	σ_timestep = ceil(options.dt*σ_s)
	h = causal_gaussian_impulse_response(σ_timestep, maxtimesteps)
	w = conv(ones(maxtimesteps),h)[1:maxtimesteps]
	nneurons = length(𝐄[1].spiketrain_leftchoice)
	map(1:nneurons) do n
		map(𝐄) do E
			conv_causal_gaussian(h, w, E.spiketrain_leftchoice[n])
		end
	end

	psth = (zeros(maxtimesteps) for n = 1:nneurons)
	psth_leftchoice = (zeros(maxtimesteps) for n = 1:nneurons)
	psth_rightchoice = (zeros(maxtimesteps) for n = 1:nneurons)
	weights_leftchoice = zeros(maxtimesteps)
	weights_rightchoice = zeros(maxtimesteps)
	for E in 𝐄
		for t in eachindex(E.spiketrain_leftchoice[1])
			weights_leftchoice += 1-E.rightchoice
			for n = 1:nneurons
				psth_leftchoice[n][t] += E.spiketrain_leftchoice[n][t]
				psth_rightchoice[n][t] += E.spiketrain_rightchoice[n][t]
				psth[n][t] += E.spiketrain_leftchoice[n][t]*(1-E.rightchoice) + E.spiketrain_rightchoice[n][t]*E.rightchoice
			end
		end
	end
f = Normal()
v = collect(pdf(f, x) for x in 0:0.1:1)
conv(u,v)./conv(ones(length(u)), v)
end

"""

Weight convolution of a spike train with a causal gaussian

Spike train responses near the beginning are adjusted for the lack of responses available before the beginning

ARGUMENT
-`𝐲`: spike trains of a neuron in each trial
-`σ`: standard deviation of the causal gaussian filter, in time steps

RETURN
-`psth`: a nested array whose element `psth[m][t]` correspons to the smoothed
"""
function convolve_causal_gaussian(𝐲::Vector{<:Vector{<:Real}}, σ::Real)
	maxtimesteps = maximum(length(𝐲ₘ) for 𝐲ₘ in 𝐲)
	normal = Normal(0,σ)
	𝐯 = collect(pdf(normal,t) for t=1:maxtimesteps)
	𝐰 = conv(ones(maxtimesteps),𝐯)[1:maxtimesteps]
	map(𝐲) do 𝐲ₘ
		𝐱 = conv(𝐲ₘ, 𝐯)
		collect(𝐱[i]/𝐰[i] for i in eachindex(𝐲ₘ))
	end
end
