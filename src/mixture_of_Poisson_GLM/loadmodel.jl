
"""
	initialize_mpGLMs(options, trials)

Initialize the Poisson mixture generalized linear model for each neuron in a trialset

ARGUMENT
-`options`: a composite containing the fixed hyperparameters of the model
-`trials`: a vector of structs each of which containing the data of one trial

RETURN
-a vector of structs each of which corresponds to the Poisson mixture generalized linear model of a neuron
"""
function initialize_mpGLMs(options::Options, trials::Vector{<:Trial})
	Φpoststereoclick = temporal_basis_functions("poststereoclick", options)
	Φpostspike = temporal_basis_functions("postspike", options)
	Φpremovement = temporal_basis_functions("premovement", options)
	movementtimesteps = collect(trial.movementtimestep for trial in trials)
	trialdurations = collect(trial.ntimesteps for trial in trials)
	𝐔poststereoclick = poststereoclickbasis(Φpoststereoclick, trialdurations)
	𝐔premovement = premovementbasis(movementtimesteps, Φpremovement, trialdurations)
	photostimulus_incline_on_s = collect(trial.photostimulus_incline_on_s for trial in trials)
	photostimulus_decline_on_s = collect(trial.photostimulus_decline_on_s for trial in trials)
	Φpostphotostimulus, Φpostphotostimulus_timesteps, 𝐔postphotostimulus = photostimulusbasis(options, photostimulus_incline_on_s, photostimulus_decline_on_s, trialdurations)
	d𝛏_dB=(2collect(1:options.Ξ) .- options.Ξ .- 1)./(options.Ξ-1)
	stereoclick_times_s = collect(trial.stereoclick_time_s for trial in trials)
	nneurons = length(trials[1].spiketrains)
	map(1:nneurons) do n
		𝐲 = vcat((trial.spiketrains[n] for trial in trials)...)
		T = sum(trialdurations)
		@assert length(𝐲)==T
		Φgain, 𝐔gain = drift_design_matrix(options, stereoclick_times_s, trialdurations, 𝐲)
		𝐔postspike = spikehistorybasis(Φpostspike, trialdurations, 𝐲)
		𝐗=hcat(𝐔gain, 𝐔postspike, 𝐔poststereoclick, 𝐔premovement, 𝐔postphotostimulus, fill(options.sf_mpGLM[1], T))
		indices𝐮 = Indices𝐮(size(𝐔gain,2), size(Φpostspike,2), size(Φpoststereoclick,2), size(Φpremovement,2), size(Φpostphotostimulus,2))
		glmθ = GLMθ(indices𝐮, options)
		MixturePoissonGLM(Δt=options.Δt,
						d𝛏_dB=d𝛏_dB,
						Φgain=Φgain,
						Φpostphotostimulus=Φpostphotostimulus,
						Φpostphotostimulus_timesteps=Φpostphotostimulus_timesteps,
						Φpostspike=Φpostspike,
						Φpoststereoclick=Φpoststereoclick,
						Φpremovement=Φpremovement,
						sf_mpGLM = options.sf_mpGLM[1],
						sf_y = options.sf_y,
						θ=glmθ,
						𝐗=𝐗,
						𝐲=𝐲)
	end
end

"""
	GLMθ(indices𝐮, options)

Randomly initiate the parameters for a mixture of Poisson generalized linear model

ARGUMENT
-`indices𝐮`: indices of the encoding weights of the temporal basis vectors of each filter that is independent of the accumulator
-`options`: settings of the model

OUTPUT
-an instance of `GLMθ`
"""
function GLMθ(indices𝐮::Indices𝐮, options::Options)
	n𝐮 = maximum(vcat((getfield(indices𝐮, field) for field in fieldnames(Indices𝐮))...))
	θ = GLMθ(fit_b = options.fit_b,
			 fit_c = options.fit_c,
			 fit_β = options.fit_β,
		 	 𝐮 = fill(NaN, n𝐮),
			 indices𝐮=indices𝐮)
	randomizeparameters!(θ, options)
	return θ
end

"""
	randomizeparameters!(glmθ, options)

Randomly initialize parameters of a mixture of Poisson GLM

MODIFIED ARGUMENT
-`θ`: structure containing parameters of a mixture of Poisson GLM

UNMODIFIED ARGUMENT
-`options`: hyperparameters of the model
"""
function randomizeparameters!(glmθ::GLMθ, options::Options)
	glmθ.b[1] = 0.0
	for i in eachindex(glmθ.𝐮)
		glmθ.𝐮[i] = 1.0 - 2rand()
	end
	for fieldname in fieldnames(typeof(glmθ.indices𝐮))
		indices = getfield(glmθ.indices𝐮, fieldname)
		glmθ.𝐮[indices] ./= options.sf_mpGLM[1]
	end
    glmθ.v[1] = (1.0 - 2rand())/options.sf_mpGLM[1]
	if glmθ.fit_β
		glmθ.β[1] = -glmθ.v[1]
	end
end
