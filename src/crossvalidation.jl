"""
    crosssvalidate(model)

Assess how well the factorial hidden Markov drift-diffusion model generalizes to independent datasets

ARGUMENT
-`kfold`: number of cross-validation folds
-`model`: a structure containing the settings, data, and parameters of a factorial hidden-Markov drift-diffusion model

OPTIONAL ARGUMENT
-`choicesonly`: whether to train on only the behavioral choices and ignore the spike trains

OUTPUT
-an instance of `CVResults`
"""
function crossvalidate(kfold::Integer, model::Model; choicesonly::Bool=false)
    cvindices = CVIndices(model, kfold)
	trainingmodels = map(cvindices->train(cvindices, model;choicesonly=choicesonly), cvindices)
	trainingsummaries = collect(Summary(trainingmodel) for trainingmodel in trainingmodels)
	testmodels = collect(test(cvindex, model, trainingmodel) for (cvindex, trainingmodel) in zip(cvindices, trainingmodels))
	rll_choice, rll_spikes = relative_loglikelihood(cvindices, testmodels, trainingmodels)
	predictions = Predictions(cvindices, testmodels)
    CVResults(cvindices=cvindices, predictions=predictions, rll_choice=rll_choice, rll_spikes=rll_spikes, trainingsummaries=trainingsummaries)
end

"""
	train(cvindices, model)

Fit training models

ARGUMENT
-`cvindices`: indices of the trials and timesteps used for training and testing in each fold
-`model`: structure containing the full dataset, parameters, and hyperparameters

OPTIONAL ARGUMENT
-`choicesonly`: whether to train on only the behavioral choices and ignore the spike trains

RETURN
-`trainingmodel`: structure containing the data in the training trials, parameters optimized for the data in the trainings, and hyperparameters
"""
function train(cvindices::CVIndices, model::Model; choicesonly::Bool=false)
	θ₀native = FHMDDM.randomize_latent_parameters(model.options)
	training_trialsets = trainingset(cvindices, model.trialsets)
	gaussianprior = GaussianPrior(model.options, training_trialsets)
	trainingmodel = Model(trialsets = training_trialsets,
						  options = model.options,
						  gaussianprior = gaussianprior,
						  θ₀native = θ₀native,
						  θnative = FHMDDM.copy(θ₀native),
						  θreal = native2real(model.options, θ₀native))
	if choicesonly
		fitonlychoices!(trainingmodel)
	else
		learnparameters!(trainingmodel)
	end
	return trainingmodel
end

"""
	relative_loglikelihood(cvindices, testmodels, trainingmodels)

Relative log-likelihood of choices and spike trains

ARGUMENT
-`cvindices`: indices of the trials and timesteps used for training and testing in each fold
-`testmodels`: a vector of structures containing the hold-out data and parameters fitted to training data for each cross-validation fold
-`trainingmodels`: a vector of structures containing the training data and parameters fitted to those data for each cross-validation fold

OUTPUT
-`rll_choice`: The element `rll_choice[i][m]` corresponds to the log-likelihood(base 2) of the choice in the m-th trial in the i-th trialset, relative to the log-likelihood predicted by a null model of choice. The null model is a Bernoulli parametrized the fraction of right choices in the training data.
-`rll_spikes`: The element `rll_spikes[i][n]` corresponds to the log-likelihood(base 2) of the spike train of the n-th neuron in the i-th trialset, relative to the log-likelihood predicted by a homogenous Poisson whose intensity is inferred from the mean response of the training data of the same neuron. The relative log-likelihood of each spike train is furthermore divided by the total number of spikes emitted by the neuron.
"""
function relative_loglikelihood(cvindices::Vector{<:CVIndices}, testmodels::Vector{<:Model}, trainingmodels::Vector{<:Model})
	ntrials_per_trialset = collect(sum(testmodel.trialsets[i].ntrials for testmodel in testmodels) for i in eachindex(testmodels[1].trialsets))
	rll_choice = collect(fill(NaN, ntrials) for ntrials in ntrials_per_trialset)
	rll_spikes = collect(zeros(length(trialset.mpGLMs)) for trialset in testmodels[1].trialsets)
	for (cvindices, testmodel, trainingmodel) in zip(cvindices, testmodels, trainingmodels)
		relative_loglikelihood!(rll_choice, rll_spikes, cvindices, testmodel, trainingmodel)
	end
	for i in eachindex(rll_spikes)
		for n in eachindex(rll_spikes[i])
			nspikes = 0
			for testmodel in testmodels
				nspikes += sum(testmodel.trialsets[i].mpGLMs[n].𝐲)
			end
			rll_spikes[i][n] /= nspikes
		end
	end
	return rll_choice, rll_spikes
end

"""
	relative_loglikelihood(rll_choice, rll_spikes, cvindices, testmodel, trainingmodel)

In-place computation of relative log-likelihoods for one cross-validation fold

MODIFIED ARGUMENTS
-`rll_choice`: see above
-`rll_spikes`: see above

UNMODIFIED ARGUMENTS
-`cvindices`: index of the trials and timesteps used for training and testing in a single fold
-`testmodel`: a structure containing the hold-out data and parameters fitted to training data for one cross-validation fold
-`trainingmodel`: a structure containing the training data and parameters fitted to those data for one cross-validation fold
"""
function relative_loglikelihood!(rll_choice::Vector{<:Vector{<:AbstractFloat}}, rll_spikes::Vector{<:Vector{<:AbstractFloat}}, cvindices::CVIndices, testmodel::Model, trainingmodel::Model)
	ℓ𝑑, ℓ𝑦 = relative_loglikelihood(testmodel, trainingmodel)
	for i in eachindex(testmodel.trialsets)
		rll_choice[i][cvindices.testingtrials[i]] .= ℓ𝑑[i]
		for n in eachindex(testmodel.trialsets[i].mpGLMs)
			rll_spikes[i][n] += ℓ𝑦[i][n]
		end
	end
	return nothing
end

"""
	relative_loglikelihood(testmodel, trainingmodel)

Compute the relative log-likelihood of the choices and spike train responses

ARGUMENT
-`testmodel`: a structure containing only the test data and the parameters learned from the training data
-`trainingmodel`: a structure containing only the training data and the parameters learned from the training data

OUTPUT
-`ℓ𝑑`: ℓ𝑑[i][m] corresponds to the log-likelihood(base 2) of the choice in the m-th trial in the i-th trialset, relative to the log-likelihood predicted by a Bernoulli model whose parameter is inferred from the fraction of right choices in the training data
-`ℓ𝑦`: ℓ𝑦[i][n] corresponds to the log-likelihood(base 2) of the spiking of the n-th neuron in the i-th trialset, relative to the log-likelihood predicted by a homogenous Poisson whose intensity is inferred from the mean response of the training data of the same neuron.
"""
function relative_loglikelihood(testmodel::Model, trainingmodel::Model)
	choices_across_trialsets = vcat((collect(trial.choice for trial in trialset.trials) for trialset in trainingmodel.trialsets)...)
	probabilityright = mean(choices_across_trialsets)
	nullchoicemodel = Bernoulli(probabilityright)
	nullspikemodels = map(trainingmodel.trialsets) do trialset
				map(trialset.mpGLMs) do mpGLM
					Poisson(mean(mpGLM.𝐲))
				end
			end
	memory = Memoryforgradient(testmodel)
	@unpack Aᵃinput, Aᵃsilent, Aᶜ, πᶜ, K, Ξ = memory
	@unpack θnative = testmodel
	P = update!(memory, testmodel)
	ℓ𝑑 = map(trialset->fill(NaN, length(trialset.trials)), testmodel.trialsets)
	ℓ𝑦 = map(trialset->zeros(length(trialset.mpGLMs)), testmodel.trialsets)
	p𝑦 = zeros(Ξ,K)
	p𝑑_a = zeros(Ξ)
	log2e = log2(exp(1))
	for i in eachindex(testmodel.trialsets)
		trialset = testmodel.trialsets[i]
		τ = 0
		for m in eachindex(trialset.trials)
			@unpack choice, clicks, ntimesteps, previousanswer = trialset.trials[m]
			if length(clicks.time) > 0
				adaptedclicks = adapt(clicks, θnative.k[1], θnative.ϕ[1])
			end
			priorprobability!(P, previousanswer)
			p𝐚 = copy(P.𝛑)
			p𝐜 = πᶜ
			for t=1:ntimesteps
				τ+=1
				if t > 1
					if isempty(clicks.inputindex[t])
						Aᵃ = Aᵃsilent
					else
						Aᵃ = Aᵃinput[clicks.inputindex[t][1]]
						update_for_transition_probabilities!(P, adaptedclicks, clicks, t)
						transitionmatrix!(Aᵃ, P)
					end
					p𝐚 = Aᵃ*p𝐚
					p𝐜 = Aᶜ*p𝐜
				end
				for n in eachindex(trialset.mpGLMs)
					conditionallikelihood!(p𝑦, trialset.mpGLMs[n], τ)
					ℓ𝑦[i][n] += log2(p𝐚'*p𝑦*p𝐜) - log2e*Distributions.logpdf(nullspikemodels[i][n], trialset.mpGLMs[n].𝐲[τ])
				end
			end
			conditionallikelihood!(p𝑑_a, choice, θnative.ψ[1])
			ℓ𝑑[i][m] = log2(p𝑑_a⋅p𝐚) - log2e*Distributions.logpdf(nullchoicemodel, choice)
		end
	end
	return ℓ𝑑, ℓ𝑦
end

"""
    CVIndices(model, kfold)

Create indices for cross-validation

ARGUMENT
-`model`: a structure containing the settings, data, and parameters of a factorial hidden-Markov drift-diffusion model
-`kfold`: number of cross-validation folds

OUTPUT
-a vector of instances of `CVIndices`
"""
function CVIndices(model::Model, kfold::integertype) where {integertype<:Integer}
    testingtrials = map(k->map(trialset->integertype[], model.trialsets), 1:kfold)
    trainingtrials = map(k->map(trialset->integertype[], model.trialsets), 1:kfold)
    testingtimesteps = map(k->map(trialset->integertype[], model.trialsets), 1:kfold)
    trainingtimesteps = map(k->map(trialset->integertype[], model.trialsets), 1:kfold)
    for i in eachindex(model.trialsets)
        ntrials = length(model.trialsets[i].trials)
        ntestingtrials = cld(ntrials, kfold)
        partitionedtrials = collect(Base.Iterators.partition(shuffle(1:ntrials), ntestingtrials))
        timesteps = map(trial->collect(1:trial.ntimesteps), model.trialsets[i].trials)
        count = 0
        for m in eachindex( model.trialsets[i].trials)
            timesteps[m] .+= count
            count +=  model.trialsets[i].trials[m].ntimesteps
        end
        for k=1:kfold
            testingtrials[k][i] = sort(partitionedtrials[k])
            trainingtrials[k][i] = sort(vcat(partitionedtrials[vcat(1:k-1, k+1:kfold)]...))
            testingtimesteps[k][i] =  vcat(timesteps[testingtrials[k][i]]...)
            trainingtimesteps[k][i] =  vcat(timesteps[trainingtrials[k][i]]...)
        end
    end
    map(1:kfold,
        trainingtrials,
        testingtrials,
        trainingtimesteps,
        testingtimesteps) do    k,
                                trainingtrials,
                                testingtrials,
                                trainingtimesteps,
                                testingtimesteps
        CVIndices(trainingtrials = trainingtrials,
                  testingtrials = testingtrials,
                  trainingtimesteps = trainingtimesteps,
                  testingtimesteps = testingtimesteps)
    end
end

"""
    trainingset(cvindices, trialsets)

Subsample the data for training

ARGUMENT
-`cvindices`: an instance of `CVIndices`
-`trialsets`: a vector of instances of `Trialset`

OUTPUT
- a vector of instances of `Trialset`
"""
function trainingset(cvindices::CVIndices, trialsets::Vector{<:Trialset})
    map(trialsets, cvindices.trainingtrials, cvindices.trainingtimesteps) do trialset, trainingtrials, trainingtimesteps
        subsample(trialset, trainingtrials, trainingtimesteps)
    end
end

"""
	test(cvindices, model, trainingmodel)

Construct a model with only the test data and the parameters learned from the training data

ARGUMENT
-`cvindices`: indices of the trials and timesteps used for training and testing in each fold
-`model`: structure containing both the test and training data
-`trainingmodel`: structure containing only the training data and the parameters learned for those data

OUTPUT
-`testmodel`: a structure containing only the test data and the parameters learned from the training data
"""
function test(cvindices::CVIndices, model::Model, trainingmodel::Model)
    testtrialsets = testingset(cvindices, model.trialsets)
	for i in eachindex(testtrialsets)
		for n in eachindex(testtrialsets[i].mpGLMs)
			sortparameters!(testtrialsets[i].mpGLMs[n].θ, trainingmodel.trialsets[i].mpGLMs[n].θ)
		end
	end
	Model(trialsets = testtrialsets,
		options = model.options,
		gaussianprior = trainingmodel.gaussianprior,
		θ₀native = trainingmodel.θ₀native,
		θnative = trainingmodel.θnative,
		θreal = native2real(model.options, trainingmodel.θnative))
end

"""
    testingset(cvindices, trialsets)

Subsample the data for testing

ARGUMENT
-`cvindices`: an instance of `CVIndices`
-`trialsets`: a vector of instances of `Trialset`

OUTPUT
- a vector of instances of `Trialset`
"""
function testingset(cvindices::CVIndices, trialsets::Vector{<:Trialset})
    map(trialsets, cvindices.testingtrials, cvindices.testingtimesteps) do trialset, testingtrials, testingtimesteps
        subsample(trialset, testingtrials, testingtimesteps)
    end
end

"""
    subsample(trialset, trialindices, timesteps)

Create an instance of `Trialset` by subsampling

ARGUMENT
-`trialset`: a structure corresponding to a collection of trials in which the same neurons werre recorded
-`trialindices`: a vector of integers indexing the trials to include
-`timesteps`: a vector of integers indexing the timesteps of spiketrains to include

OUTPUT
- an instance of `Trialset`
"""
function subsample(trialset::Trialset, trialindices::Vector{<:Integer}, timesteps::Vector{<:Integer})
	trials = trialset.trials[trialindices]
	𝛕₀ = cumsum(vcat(0, collect(trials[m].ntimesteps for m=1:length(trials)-1)))
	trials = collect(FHMDDM.reindex(index_in_trialset, τ₀, trial) for (index_in_trialset, τ₀, trial) in zip(1:length(trials), 𝛕₀, trials))
    Trialset(trials = trials, mpGLMs = map(mpGLM->subsample(mpGLM, timesteps), trialset.mpGLMs))
end

"""
	dictionary(cvresults)

Convert an instance of `CVResults` to a dictionary
"""
function dictionary(cvresults::CVResults)
	Dict("cvindices" => map(dictionary, cvresults.cvindices),
		"predictions" => dictionary(cvresults.predictions),
		"rll_choice"=>cvresults.rll_choice,
		"rll_spikes"=>cvresults.rll_spikes,
		"trainingsummaries"=>map(dictionary, cvresults.trainingsummaries))
end

"""
	dictionary(cvindices)

Convert an instance of 'CVIndices' to a dictionary
"""
function dictionary(cvindices::CVIndices)
	Dict("testingtrials" => cvindices.testingtrials,
		 "trainingtrials" => cvindices.trainingtrials,
		 "testingtimesteps" => cvindices.testingtimesteps,
		 "trainingtimesteps" => cvindices.trainingtimesteps)
end

"""
    save(cvresults,options)

Save the results of crossvalidation

ARGUMENT
-`cvresults`: an instance of `CVResults`, a drift-diffusion linear model
"""
function save(cvresults::CVResults, options::Options; folderpath::String=dirname(options.datapath), prefix::String="cvresults")
	dict = Dict("cvindices" => map(dictionary, cvresults.cvindices),
				"rll_choice"=>cvresults.rll_choice,
				"rll_spikes"=>cvresults.rll_spikes,
				"trainingsummaries"=>map(dictionary, cvresults.trainingsummaries))
    matwrite(joinpath(folderpath, prefix*".mat"), dict)
	save(cvresults.predictions, options; folderpath=folderpath, prefix=prefix)
    return nothing
end
