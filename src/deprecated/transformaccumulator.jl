"""
    transformaccumulator

Nonlinearly transform the normalized values of the accumulator

ARGUMENT
-`ξ`: value of the accumulator: expected to be between -1 and 1
-`b`: parameter specifying the transformation

RETURN
-transformed value of the accumulator
"""
function transformaccumulator(b::Real, ξ::Real)
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
function dtransformaccumulator(b::Real, ξ::Real)
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
