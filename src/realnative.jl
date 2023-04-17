"""
	real2native(r,q,l,u)

Convert a parameter from real space to latent space

ARGUMENT
-`r`: value in real space
-`q`: value in native space equal to a zero-valued in real space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-scalar representing the value in native space
"""
function real2native(r::Real, q::Real, l::Real, u::Real)
	if q == l
		l + (u-l)*logistic(r)
	else
		l + (u-l)*logistic(r+logit((q-l)/(u-l)))
	end
end

"""
	native2real(n,q,l,u)

Convert a parameter from native space to real space

ARGUMENT
-`n`: value in native space
-`q`: value in native space equal to a zero-valued in real space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-scalar representing the value in real space
"""
function native2real(n::Real, q::Real, l::Real, u::Real)
	if q == l
		logit((n-l)/(u-l))
	else
		logit((n-l)/(u-l)) - logit((q-l)/(u-l))
	end
end

"""
	differentiate_native_wrt_real(r,q,l,u)

Derivative of the native value of a parameter with respect to its value in real space

ARGUMENT
-`r`: value in real space
-`q`: value in native space equal to a zero-valued in real space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-a scalar representing the derivative
"""
function differentiate_native_wrt_real(r::Real, q::Real, l::Real, u::Real)
	if q == l
		x = logistic(r)
	else
		x = logistic(r + logit((q-l)/(u-l)))
	end
	(u-l)*x*(1.0-x)
end

"""
	differentiate_twice_native_wrt_real(r,q,l,u)

Second derivative of the native value of a parameter with respect to its value in real space

ARGUMENT
-`r`: value in real space
-`q`: value in native space equal to a zero-valued in real space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-a scalar representing the second derivative
"""
function differentiate_twice_native_wrt_real(r::Real, q::Real, l::Real, u::Real)
	if q == l
		x = logistic(r)
	else
		x = logistic(r + logit((q-l)/(u-l)))
	end
	(u-l)*(x*(1.0-x)^2 - x^2*(1-x))
end

"""
	differentiate_native_wrt_real(r,l,u)

Derivative of the native value of hyperparameters with respect to values in real space

ARGUMENT
-`r`: value in real space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-derivatives
"""
function differentiate_native_wrt_real(r::Real, l::Real, u::Real)
	x = logistic(r)
	(u-l)*x*(1.0-x)
end
