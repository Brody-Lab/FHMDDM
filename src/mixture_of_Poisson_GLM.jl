"""
    likelihood(mpGLM, j, k)

Conditional likelihood of the spike train, given the index of the state of the accumulator `j` and the state of the coupling `k`, and also by the prior likelihood of the regression weights

UNMODIFIED ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model for one neuron
-`j`: state of accumulator variable
-`k`: state of the coupling variable

RETURN
-`𝐩`: a vector by which the conditional likelihood of the spike train and the prior likelihood of the regression weights are multiplied against
"""
function likelihood(mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack Δt, 𝐲, 𝐲! = mpGLM
    𝐗𝐰 = linearpredictor(mpGLM, j, k)
    𝐩 = 𝐗𝐰 # reuse memory
    for i=1:length(𝐩)
        λΔt = softplus(𝐗𝐰[i])*Δt
        if 𝐲[i]==0
            𝐩[i] = exp(-λΔt)
        elseif 𝐲[i]==1
            𝐩[i] = λΔt/exp(λΔt)
        else
            𝐩[i] = λΔt^𝐲[i] / exp(λΔt) / 𝐲![i]
        end
    end
    return 𝐩
end

"""
    likelihood!(𝐩, mpGLM, j, k)

In-place multiplication of `𝐩` by the conditional likelihood of the spike train, given the index of the state of the accumulator `j` and the state of the coupling `k`, and also by the prior likelihood of the regression weights

MODIFIED ARGUMENT
-`𝐩`: a vector by which the conditional likelihood of the spike train and the prior likelihood of the regression weights are multiplied against

UNMODIFIED ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model for one neuron
-`j`: state of accumulator variable
-`k`: state of the coupling variable

RETURN
-`nothing`
"""
function likelihood!(𝐩::Vector{<:Real}, mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack Δt, 𝐲, 𝐲! = mpGLM
    𝐗𝐰 = linearpredictor(mpGLM, j, k)
    for i=1:length(𝐩)
        λΔt = softplus(𝐗𝐰[i])*Δt
        if 𝐲[i]==0
            𝐩[i] *= exp(-λΔt)
        elseif 𝐲[i]==1
            𝐩[i] *= λΔt/exp(λΔt)
        else
            𝐩[i] *= λΔt^𝐲[i] / exp(λΔt) / 𝐲![i]
        end
    end
    return nothing
end

"""
    linearpredictor(mpGLM, j, k)

Linear combination of the weights in the j-th accumulator state and k-th coupling state

ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model of one neuron
-`j`: state of the accumulator variable
-`k`: state of the coupling variable

RETURN
-`𝛌`: a vector whose element 𝛌[t] corresponds to the t-th time bin in the trialset
"""
function linearpredictor(mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack 𝐔, 𝐗, 𝛏 = mpGLM
    @unpack 𝐮, 𝐯, a, b = mpGLM.θ
    if k == 1 && 𝛏[j] != 0.0
        ξ = transformaccumulator(b[1], 𝛏[j])
        if 𝛏[j] < 0
            𝐰 = vcat(𝐮, ξ.*𝐯)
        else
            𝐰 = vcat(𝐮, rectifya(a[1]).*ξ.*𝐯)
        end
        𝐗*𝐰
    else
        𝐔*𝐮
    end
end

"""
    transformaccumulator

Nonlinearly transform the normalized values of the accumulator

ARGUMENT
-`ξ`: value of the accumulator: expected to be between -1 and 1
-`b`: parameter specifying the transformation

RETURN
-transformed value of the accumulator
"""
function transformaccumulator(b::Real, ξ::AbstractFloat)
    if b == 0.0
        ξ
    else
        if ξ < 0
            if b > 709.0 # 709 is close to which exp returns Inf for a 64-bit floating point number
                ξ == -1.0 ? -1.0 : 0.0
            else
                -expm1(-b*ξ)/expm1(b)
                # (exp(-b*ξ)-1.0)/(1.0-exp(b))
            end
        elseif ξ > 0
            if b > 709.0
                ξ == 1.0 ? 1.0 : 0.0
            else
                expm1(b*ξ)/expm1(b)
                # (1.0-exp(b*ξ))/(1.0-exp(b))
            end
        else
            0.0
        end
    end
end

"""
    dtransformaccumulator

Derivative of the nonlinear transformation of the normalized values of the accumulator with respect to b

ARGUMENT
-`ξ`: value of the accumulator: expected to be between -1 and 1
-`b`: parameter specifying the transformation

RETURN
-transformed value of the accumulator
"""
function dtransformaccumulator(b::Real, ξ::AbstractFloat)
    if ξ == -1.0 || ξ == 0.0 || ξ == 1.0 || b > 709.0 # 709 is close to which exp returns Inf for a 64-bit floating point number
        0.0
    elseif abs(b) < 1e-6
        ξ < 0 ? (-ξ^2-ξ)/2 : (ξ^2-ξ)/2
    elseif ξ < 0
        eᵇ = exp(b)
        eᵇm1 = expm1(b)
        e⁻ᵇˣ = exp(-b*ξ)
        e⁻ᵇˣm1 = expm1(-b*ξ)
        if b < 1
            (ξ*e⁻ᵇˣ*eᵇm1 + e⁻ᵇˣm1*eᵇ)/eᵇm1^2
        else
            ξ*e⁻ᵇˣ/eᵇm1 + e⁻ᵇˣm1/(eᵇ-2+exp(-b))
        end
    elseif ξ > 0
        eᵇ = exp(b)
        eᵇm1 = expm1(b)
        eᵇˣ = exp(b*ξ)
        eᵇˣm1 = expm1(b*ξ)
        if b < 1
            ξ*eᵇˣ/eᵇm1 - eᵇˣm1*eᵇ/eᵇm1^2
        else
            ξ*eᵇˣ/eᵇm1 - eᵇˣm1/(eᵇ-2+exp(-b))
        end
    end
end

"""
    rectifya(a)

Map a parameter from real space to positive values

A value of 0 in real space corresponds to 1.0

ARGUMENT:
-`a`: parameter in real space

OUTPUT
-positive-valued parameter
"""
function rectifya(a::Real)
    # softplus(a+log(exp(1)-1.0))
    # 0.2 + 4.5 *logistic(a + logit(8.0/45.0))
    a+1.0
end

"""
    drectifya(a)

Derivative of the mapping of a parameter from real space to positive values

ARGUMENT:
-`a`: parameter in real space

OUTPUT
-the derivative
"""
function drectifya(a::Real)
    # logistic(a+log(exp(1)-1.0))
    # logistica = logistic(a + logit(8.0/45.0))
    # 4.5*logistica*(1.0-logistica)
    1.0
end

"""
    estimatefilters!(trialsets, γ)

Update the filters in the mixture of Poisson generalized linear models

MODIFIED ARGUMENT
-`trialsets`: vector of data for each group of trials

UNMODIFIED ARGUMENT
-`γ`: joint posterior likelihood of each accumulator state and each coupling state at each time bin. `γ[i][ξ,k][t]` corresponds to the joint posterior of accumulator state ξ and coupling state k at the t-th time bin concatenated across trials in the i-th trialset

RETURN
-nothing
"""
function estimatefilters!(trialsets::Vector{<:Trialset},
                          γ::Vector{<:Matrix{<:Vector{<:AbstractFloat}}},
                          options::Options;
                          show_trace::Bool=true)
    concatentatedθ = map(trialsets, γ) do trialset, γ
                        pmap(trialset.mpGLMs) do mpGLM
                            estimatefilters(γ, mpGLM;
                                            fit_a=options.fit_a,
                                            fit_b=options.fit_b,
                                            show_trace=show_trace)
                        end
                    end
    Pᵤ = length(trialsets[1].mpGLMs[1].θ.𝐮)
    Pᵥ = length(trialsets[1].mpGLMs[1].θ.𝐯)
    for i in eachindex(concatentatedθ)
        for n in eachindex(concatentatedθ[i])
            trialsets[i].mpGLMs[n].θ.𝐮 .= concatentatedθ[i][n][1:Pᵤ]
            trialsets[i].mpGLMs[n].θ.𝐯 .= concatentatedθ[i][n][Pᵤ+1:Pᵤ+Pᵥ]
            counter = Pᵤ+Pᵥ
            if options.fit_a
                trialsets[i].mpGLMs[n].θ.a .= concatentatedθ[i][n][counter+=1]
            end
            if options.fit_b
                trialsets[i].mpGLMs[n].θ.b .= concatentatedθ[i][n][counter+=1]
            end
        end
    end
    return nothing
end

"""
    estimatefilters(γ, mpGLM)

Estimate the filters of the Poisson mixture GLM of one neuron

ARGUMENT
-`γ`: posterior probabilities of the latent
-`mpGLM`: the Poisson mixture GLM of one neuron

OPTIONAL ARGUMENT
-`show_trace`: whether to show information about each step of the optimization
-`fit_a`: whether to fit the asymmetric scaling factor
-`fit_b`: whether to fit the nonlinearity factor

RETURN
-weights concatenated into a single vector
"""
function estimatefilters(γ::Matrix{<:Vector{<:AbstractFloat}},
                         mpGLM::MixturePoissonGLM;
                         show_trace::Bool=true,
                         fit_a::Bool=true,
                         fit_b::Bool=true)
    @unpack 𝐮, 𝐯, a, b = mpGLM.θ
    x₀ = vcat(𝐮, 𝐯)
    fit_a && (x₀ = vcat(x₀, a))
    fit_b && (x₀ = vcat(x₀, b))
    f(x) = negativeexpectation(γ, mpGLM, x; fit_a=fit_a, fit_b=fit_b)
    g!(∇, x) = ∇negativeexpectation!(∇, γ, mpGLM, x; fit_a=fit_a, fit_b=fit_b)
    # h!(𝐇, x) = 𝐇negativeexpectation!(𝐇, γ, mpGLM, x)
    # results = Optim.optimize(f, g!, h!, x₀, NewtonTrustRegion(), Optim.Options(show_trace=show_trace))
    results = Optim.optimize(f, g!, x₀, LBFGS(), Optim.Options(show_trace=show_trace))
    show_trace && println("The model converged: ", Optim.converged(results))
    return Optim.minimizer(results)
end

"""
    negativeexpectation(γ, mpGLM, x)

Negative of the expectation of the log-likelihood of the mixture of Poisson generalized model of one neuron

ARGUMENT
-`γ`: posterior probability of the latent variable
-`mpGLM`: the GLM of one neuron
-`x`: weights of the linear filters of the GLM concatenated as a vector of floating-point numbers

RETURN
-expectation of the log-likelihood of the spike train of one neuron
"""
function negativeexpectation(γ::Matrix{<:Vector{<:AbstractFloat}},
                             mpGLM::MixturePoissonGLM,
                             x::Vector{<:Real};
                             fit_a::Bool=true,
                             fit_b::Bool=true)
    @unpack Δt, K, 𝐔, 𝚽, 𝛏, 𝐗, 𝐲 = mpGLM
    Pᵤ = size(𝐔,2)
    Pᵥ = size(𝚽,2)
    𝐮 = x[1:Pᵤ]
    𝐯 = x[Pᵤ+1:Pᵤ+Pᵥ]
    counter = Pᵤ+Pᵥ
    a = fit_a ? x[counter+=1] : mpGLM.θ.a[1]
    b = fit_b ? x[counter+=1] : mpGLM.θ.b[1]
    fa = rectifya(a)
    𝐔𝐮 = 𝐔*𝐮
    T = length(𝐔𝐮)
    Ξ = size(γ,1)
    zeroindex = (Ξ+1)/2
    neg𝒬 = 0.0
    for k = 1:K
        for i = 1:Ξ
            if k == 2 || i == zeroindex
                𝐗𝐰 = 𝐔𝐮
            else
                fξ = transformaccumulator(b, 𝛏[i])
                if i < zeroindex
                    𝐰 = vcat(𝐮, fξ.*𝐯)
                else
                    𝐰 = vcat(𝐮, fa.*fξ.*𝐯)
                end
                𝐗𝐰 = 𝐗*𝐰
            end
            for t = 1:T
                λ = softplus(𝐗𝐰[t])
                if 𝐲[t] == 0
                    neg𝒬 += γ[i,k][t]*(λ*Δt)
                elseif 𝐲[t] == 1
                    neg𝒬 += γ[i,k][t]*(λ*Δt - log(λ))
                else
                    neg𝒬 += γ[i,k][t]*(λ*Δt - 𝐲[t]*log(λ))
                end
            end
        end
    end
    return neg𝒬
end

"""
    ∇negativeexpectation!(∇, γ, mpGLM, x)

Gradient of the negative of the expectation of the log-likelihood of the mixture of Poisson generalized model of one neuron

MODIFIED ARGUMENT
-`∇`: The gradient

UNMODIFIED ARGUMENT
-`γ`: Joint posterior probability of the accumulator and coupling variable. γ[i,k][t] corresponds to the i-th accumulator state and the k-th coupling state in the t-th time bin in the trialset.
-`mpGLM`: structure containing information for the mixture of Poisson GLM for one neuron
-`x`: vector of parameters for the mixture of Poisson GLM

RETURN
-nothing
"""
function ∇negativeexpectation!( ∇::Vector{<:AbstractFloat},
                                γ::Matrix{<:Vector{<:AbstractFloat}},
                                mpGLM::MixturePoissonGLM,
                                x::Vector{<:AbstractFloat};
                                fit_a::Bool=true,
                                fit_b::Bool=true)
    Pᵤ = size(mpGLM.𝐔,2)
    Pᵥ = size(mpGLM.𝚽,2)
    mpGLM.θ.𝐮 .= x[1:Pᵤ]
    mpGLM.θ.𝐯 .= x[Pᵤ+1:Pᵤ+Pᵥ]
    counter = Pᵤ+Pᵥ
    fit_a && (mpGLM.θ.a[1] = x[counter+=1])
    fit_b && (mpGLM.θ.b[1] = x[counter+=1])
    ∇ .= ∇negativeexpectation(γ, mpGLM;fit_a=fit_a, fit_b=fit_b)
    return nothing
end

"""
    ∇negativeexpectation(γ, mpGLM)

Gradient of the negative of the expectation of the log-likelihood of the mixture of Poisson generalized model of one neuron

ARGUMENT
-`γ`: Joint posterior probability of the accumulator and coupling variable. γ[i,k][t] corresponds to the i-th accumulator state and the k-th coupling state in the t-th time bin in the trialset.
-`mpGLM`: structure containing information for the mixture of Poisson GLM for one neuron

RETURN
-∇: the gradient
"""
function ∇negativeexpectation(γ::Matrix{<:Vector{<:AbstractFloat}},
                              mpGLM::MixturePoissonGLM;
                              fit_a::Bool=true,
                              fit_b::Bool=true)
    @unpack Δt, K, 𝐔, 𝚽, 𝐗, 𝛏, 𝐲 = mpGLM
    @unpack 𝐮, 𝐯, a, b = mpGLM.θ
    Ξ = size(γ,1)
    zeroindex = cld(Ξ,2)
    𝐔𝐮 = 𝐔*𝐮
    fa = rectifya(a[1])
    T = length(𝐲)
    ∑𝐮, ∑left, ∑right = zeros(T), zeros(T), zeros(T)
    fit_b && (∑b = zeros(T))
    𝛈 = 𝐔𝐮 # reuse memory
    for t in eachindex(𝛈)
        𝛈[t] = differentiate_negative_loglikelihood(Δt, 𝐔𝐮[t], 𝐲[t])
    end
    for k = 1:K
        for i = 1:Ξ
            if k == 2 || i == zeroindex
                dnegℓ = 𝛈
            else
                fξ = transformaccumulator(b[1], 𝛏[i])
                if i < zeroindex
                    𝐰 = vcat(𝐮, fξ.*𝐯)
                else
                    𝐰 = vcat(𝐮, fa.*fξ.*𝐯)
                end
                𝐗𝐰 = 𝐗*𝐰
                dnegℓ = 𝐗𝐰 # reuse memory
                for t in eachindex(dnegℓ)
                    dnegℓ[t] = differentiate_negative_loglikelihood(Δt, 𝐗𝐰[t], 𝐲[t])
                end
            end
            ζ = γ[i,k] .* dnegℓ
            ∑𝐮 .+= ζ
            if k == 1 &&  i != zeroindex
                dfξ = dtransformaccumulator(b[1], 𝛏[i])
                if i < zeroindex
                    ∑left .+= fξ.*ζ
                    fit_b && (∑b .+= dfξ.*ζ)
                elseif i > zeroindex
                    ∑right .+= fξ.*ζ
                    fit_b && (∑b .+= fa.*dfξ.*ζ)
                end
            end
        end
    end
    ∑𝐯 = ∑left # reuse memory
    ∑𝐯 .+= fa.*∑right
    𝐯ᵀ𝚽ᵀ = transpose(𝚽*𝐯)
    Pᵤ = length(𝐮)
    Pᵥ = length(𝐯)
    ∇ = zeros(Pᵤ+Pᵥ+fit_a+fit_b)
    ∇[1:Pᵤ] = transpose(𝐔)*∑𝐮
    ∇[Pᵤ+1:Pᵤ+Pᵥ] = transpose(𝚽)*∑𝐯
    counter = Pᵤ+Pᵥ
    if fit_a
        ∇[counter+=1] = drectifya(a[1])*(𝐯ᵀ𝚽ᵀ*∑right) # the parentheses avoid unnecessary memory allocation
    end
    if fit_b
        ∇[counter+=1] = 𝐯ᵀ𝚽ᵀ*∑b
    end
    return ∇
end

"""
    differentiate_negative_loglikelihood

Differentiate the negative of the log-likelihood of a Poisson GLM with respect to the linear predictor

The Poisson GLM is assumed to have a a softplus nonlinearity

ARGUMENT
-`Δt`: duration of time step
-`xw`: linear predictor at one time step
-`y`: observation at that time step

RETURN
-the derivative with respect to the linear predictor
"""
function differentiate_negative_loglikelihood(Δt::AbstractFloat, xw::Real, y::AbstractFloat)
    if y == 0
        xw < -100 ? 0.0 : logistic(xw)*Δt
    elseif y == 1
        xw < -100 ? -1.0 : logistic(xw)*(Δt - 1.0/softplus(xw))
    else
        xw < -100 ? -y : logistic(xw)*(Δt - y/softplus(xw))
    end
end

"""
    𝐇negativeexpection(𝐇, γ, mpGLM, x)

Compute the Hessian of the negative of the terms in the expectation that depend on the GLM filters

MODIFIED ARGUMENT
-`𝐇`: Hessian matrix

UNMODIFIED ARGUMENT
-`γ`: posterior probabilities of the latents
-`mpGLM`: the Poisson mixture GLM of one neuron
-`x`: filters of the Poisson mixture GLM

RETURN
-nothing
"""
function 𝐇negativeexpectation!(𝐇::Matrix{<:AbstractFloat},
                               γ::Matrix{<:Vector{<:AbstractFloat}},
                               mpGLM::MixturePoissonGLM,
                               x::Vector{<:AbstractFloat})
    @unpack Δt, 𝐔, 𝚽, 𝛏, 𝐗, 𝐲 = mpGLM
    Pᵤ = size(𝐔,2)
    Pₗ = size(𝚽,2)
    indices𝐮 = 1:Pᵤ
    indices𝐥 = Pᵤ+1:Pᵤ+Pₗ
    indices𝐫 = Pᵤ+Pₗ+1:Pᵤ+2Pₗ
    𝐮 = x[indices𝐮]
    𝐥 = x[indices𝐥]
    𝐫 = x[indices𝐫]
    𝐔𝐮 = 𝐔*𝐮
    Ξ = size(γ,1)
    zeroindex = cld(Ξ,2)
    if size(γ,2) > 1 # i.e, the coupling variable has more than one state
        ∑γdecoupled = γ[zeroindex,1] .+ sum(γ[:,2])
    else
        ∑γdecoupled = γ[zeroindex,1]
    end
    f₀ = softplus.(𝐔𝐮)
    f₁ = logistic.(𝐔𝐮) # first derivative
    f₂ = f₁ .* (1.0 .- f₁) # second derivative
    tmp𝐮𝐮 = ∑γdecoupled .* (f₂.*Δt .- 𝐲.*(f₀.*f₂ .- f₁.^2)./f₀.^2)
    T = length(𝐲)
    tmp𝐥𝐮 = zeros(T)
    tmp𝐫𝐮 = zeros(T)
    tmp𝐥𝐥 = zeros(T)
    tmp𝐫𝐫 = zeros(T)
    for i = 1:Ξ
        if i < zeroindex
            𝐗𝐰 = 𝐗*vcat(𝐮, 𝛏[i].*𝐥)
        elseif i > zeroindex
            𝐗𝐰 = 𝐗*vcat(𝐮, 𝛏[i].*𝐫)
        else
            continue
        end
        f₀ = softplus.(𝐗𝐰)
        f₁ = logistic.(𝐗𝐰) # first derivative
        f₂ = f₁ .* (1.0 .- f₁) # second derivative
        tmp = γ[i,1].*(f₂.*Δt .- 𝐲.*(f₀.*f₂ .- f₁.^2)./f₀.^2)
        if i < zeroindex
            tmp𝐥𝐥 .+= 𝛏[i]^2 .* tmp
            tmp𝐥𝐮 .+= 𝛏[i].*tmp
        elseif i > zeroindex
            tmp𝐫𝐫 .+= 𝛏[i]^2 .* tmp
            tmp𝐫𝐮 .+= 𝛏[i].*tmp
        end
        tmp𝐮𝐮 .+= tmp
    end
    𝐔ᵀ = transpose(𝐔)
    𝚽ᵀ = transpose(𝚽)
    𝐔ᵀ_tmp𝐥𝐮_𝚽 = 𝐔ᵀ*(tmp𝐥𝐮.*𝚽)
    𝐔ᵀ_tmp𝐫𝐮_𝚽 = 𝐔ᵀ*(tmp𝐫𝐮.*𝚽)
    𝐇 .= 0
    𝐇[indices𝐮, indices𝐮] = 𝐔ᵀ*(tmp𝐮𝐮.*𝐔)
    𝐇[indices𝐥, indices𝐥] = 𝚽ᵀ*(tmp𝐥𝐥.*𝚽)
    𝐇[indices𝐫, indices𝐫] = 𝚽ᵀ*(tmp𝐫𝐫.*𝚽)
    𝐇[indices𝐮, indices𝐥] = 𝐔ᵀ_tmp𝐥𝐮_𝚽
    𝐇[indices𝐮, indices𝐫] = 𝐔ᵀ_tmp𝐫𝐮_𝚽
    𝐇[indices𝐥, indices𝐮] = transpose(𝐔ᵀ_tmp𝐥𝐮_𝚽)
    𝐇[indices𝐫, indices𝐮] = transpose(𝐔ᵀ_tmp𝐫𝐮_𝚽)
    return nothing
end
