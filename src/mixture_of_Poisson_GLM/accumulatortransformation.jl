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
transformaccumulator(mpGLM::MixturePoissonGLM) = map(dξᵢ_dB->transformaccumulator(mpGLM.θ.b[1], mpGLM.sf_mpGLM, dξᵢ_dB), mpGLM.d𝛏_dB)

"""
	dtransformaccumulator(mpGLM)

Derivative of the transformation of accumulated evidence w.r.t the nonlinearity parameter

ARGUMENT
-`mpGLM`: a mixture Poisson generalized linear model

RETURN
-a vector representing the derivative for each discrete value of normalized accumulated evidence
"""
dtransformaccumulator(mpGLM::MixturePoissonGLM) = map(dξᵢ_dB->dtransformaccumulator(mpGLM.θ.b[1], mpGLM.sf_mpGLM, dξᵢ_dB), mpGLM.d𝛏_dB)

"""
	dtransformaccumulator(mpGLM)

Second derivative of the transformation of accumulated evidence w.r.t the nonlinearity parameter

ARGUMENT
-`mpGLM`: a mixture Poisson generalized linear model

RETURN
-a vector representing the derivative for each discrete value of normalized accumulated evidence
"""
d²transformaccumulator(mpGLM::MixturePoissonGLM) = map(dξᵢ_dB->d²transformaccumulator(mpGLM.θ.b[1], mpGLM.sf_mpGLM, dξᵢ_dB), mpGLM.d𝛏_dB)
