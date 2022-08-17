"""
Transformation and derivatives with a scale factor `s`
"""
transformaccumulator(b::Real, s::Real, ξ::Real) = transformaccumulator(b*s, ξ)
dtransformaccumulator(b::Real, s::Real, ξ::Real) = s*dtransformaccumulator(b*s, ξ)
d²transformaccumulator(b::Real, s::Real, ξ::Real) = s^2*d²transformaccumulator(b*s, ξ)

"""
	transformaccumulator(mpGLM)

Transform the values of accumulated evidence

ARGUMENT
-`mpGLM`: a mixture Poisson generalized linear model

RETURN
-a vector representing the transformed values of accumulated evidence
"""
function transformaccumulator(mpGLM::MixturePoissonGLM)
	@unpack d𝛏_dB = mpGLM
	@unpack b, b_scalefactor = mpGLM.θ
	if length(b) > 0
		map(dξᵢ_dB->transformaccumulator(b[1], b_scalefactor, dξᵢ_dB), d𝛏_dB)
	else
		d𝛏_dB
	end
end

"""
	dtransformaccumulator(mpGLM)

Derivative of the transformation of accumulated evidence w.r.t the nonlinearity parameter

ARGUMENT
-`mpGLM`: a mixture Poisson generalized linear model

RETURN
-a vector representing the derivative for each discrete value of normalized accumulated evidence
"""
function dtransformaccumulator(mpGLM::MixturePoissonGLM)
	@unpack d𝛏_dB = mpGLM
	@unpack b, b_scalefactor = mpGLM.θ
	if length(b) > 0
		d𝛚_db = map(dξᵢ_dB->dtransformaccumulator(b[1], b_scalefactor, dξᵢ_dB), d𝛏_dB)
	else
		d𝛚_db = map(dξᵢ_dB->dtransformaccumulator(0.0, dξᵢ_dB), d𝛏_dB)
	end
end

"""
	dtransformaccumulator(mpGLM)

Second derivative of the transformation of accumulated evidence w.r.t the nonlinearity parameter

ARGUMENT
-`mpGLM`: a mixture Poisson generalized linear model

RETURN
-a vector representing the derivative for each discrete value of normalized accumulated evidence
"""
function d²transformaccumulator(mpGLM::MixturePoissonGLM)
	@unpack d𝛏_dB = mpGLM
	@unpack b, b_scalefactor = mpGLM.θ
	if length(b) > 0
		d²𝛚_db² = map(dξᵢ_dB->d²transformaccumulator(b[1], b_scalefactor, dξᵢ_dB), d𝛏_dB)
	else
		d²𝛚_db² = map(dξᵢ_dB->d²transformaccumulator(0.0, dξᵢ_dB), d𝛏_dB)
	end
end
