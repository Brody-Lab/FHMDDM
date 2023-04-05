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
<<<<<<< Updated upstream
	trainingmodels = map(cvindices->train(cvindices, model;choicesonly=choicesonly), cvindices)
	trainingsummaries = collect(ModelSummary(trainingmodel) for trainingmodel in trainingmodels)
	testmodels = collect(test(cvindex, model, trainingmodel) for (cvindex, trainingmodel) in zip(cvindices, trainingmodels))
	characterization = Characterization(cvindices, testmodels, trainingmodels)
    CVResults(cvindices=cvindices, characterization=characterization, trainingsummaries=trainingsummaries)
=======
	trainingmodels = pmap(cvindices->train(cvindices, model;choicesonly=choicesonly), cvindices)
	trainingsummaries = collect(ModelSummary(trainingmodel) for trainingmodel in trainingmodels)
	testmodels = collect(test(cvindex, model, trainingmodel) for (cvindex, trainingmodel) in zip(cvindices, trainingmodels))
	characterization = Characterization(cvindices, testmodels, trainingmodels)
	psthsets = poststereoclick_time_histogram_sets(characterization.expectedemissions, model)
    CVResults(cvindices=cvindices, characterization=characterization, psthsets=psthsets, trainingsummaries=trainingsummaries)
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    Trialset(trials = trials, mpGLMs = map(mpGLM->subsample(mpGLM, timesteps), trialset.mpGLMs))
end

"""
	dictionary(cvresults)

Convert an instance of `CVResults` to a dictionary
"""
function dictionary(cvresults::CVResults)
	Dict("cvindices" => map(dictionary, cvresults.cvindices),
		"characterization" => dictionary(cvresults.characterization),
		"trainingsummaries"=>map(dictionary, cvresults.trainingsummaries))
=======
    Trialset(trials = trials, mpGLMs = map(mpGLM->subsample(mpGLM, timesteps, trialindices), trialset.mpGLMs))
>>>>>>> Stashed changes
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
    save(cvresults, folderpath)

Save the results of cross-validation

Each field of the composite `cvresults` is saved within a separate file, with the same name as that of the field, within a folder whose absolute path is specified by `folderpath`.
"""
function save(cvresults::CVResults, folderpath::String)
	if !isdir(folderpath)
		mkdir(folderpath)
		@assert isdir(folderpath)
	end
	for fieldname in (:cvindices, :trainingsummaries)
		dict = Dict(String(fieldname) => map(dictionary, getfield(cvresults, fieldname)))
		filepath = joinpath(folderpath, String(fieldname)*".mat")
	    matwrite(filepath, dict)
	end
	save(cvresults.characterization, folderpath)
<<<<<<< Updated upstream
=======
	save(cvresults.psthsets, folderpath)
>>>>>>> Stashed changes
    return nothing
end
