using FFTW
using Plots
using LinearAlgebra
using Statistics

# Inviscid Burgers: L(u) = -0.5 * (u^2)_x
function L_inviscid(u, Dx)
    return -0.5 .* (Dx * (u .^ 2))
end

# Viscous Burgers: L(u) = -0.5*(u^2)_x + eps*u_xx
function L_viscous(u, Dx, Dxx, eps)
    return -0.5 .* (Dx * (u .^ 2)) .+ eps .* (Dxx * u)
end

function matrix_form(N)
    k = fftfreq(N, N)
    Iden = Matrix(I, N, N)
    F = fft(Iden, 1)
    iF = ifft(Iden, 1)
    K = Diagonal(im .* k)
    K2 = Diagonal(-(k .^ 2))
    Dx = real.(iF * K  * F)
    Dxx = real.(iF * K2 * F)
    return Dx, Dxx
end

function ssprk33(N, u, a, L_operator, dt; filter=nothing)
    #Shu-osher RK3
    u1 = u .+ dt .* L_operator(N, u, a)
    isnothing(filter) || (u1 = apply_filter(u1, filter))
    u2 = 0.75 .* u .+ 0.25 .* (u1 .+ dt .* L_operator(N, u1, a))
    isnothing(filter) || (u2 = apply_filter(u2, filter))
    un = (1/3) .* u .+ (2/3) .* (u2 .+ dt .* L_operator(N, u2, a))
    isnothing(filter) || (un = apply_filter(un, filter))
    return un
end

function rk4(N, u, a, L_operator, dt; filter=nothing)
    k1 = L_operator(N, u, a)
    k2 = L_operator(N, u .+ 0.5 .* dt .* k1, a)
    k3 = L_operator(N, u .+ 0.5 .* dt .* k2, a)
    k4 = L_operator(N, u .+ dt .* k3, a)
    un = u .+ (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
    isnothing(filter) || (un = apply_filter(un, filter))
    return un
end

#marching forward in time
function time_forward(u0, L_operator, T, dt, N, a, evol_method; filter=nothing, record_every=10)
    u = copy(u0)
    nsteps = round(Int, T/dt)
    dt_real = T / nsteps
    fft_history = Vector{Vector{Float64}}()
    time_history = Float64[]
    for i in 1:nsteps
        u = evol_method(N, u, a, L_operator, dt; filter=filter)
        # stop if blowing up
        if any(isnan, u) || maximum(abs, u) > 1e6
            @warn "Blowup detected at step $i / $nsteps"
            break
        end
 
        if i % record_every == 0
            # store one-sided amplitude spectrum (k = 0 … N/2)
            uh = abs.(fft(u))
            push!(fft_history,  uh[1 : length(u)÷2+1])
            push!(time_history, i * dt_real)
        end
    end
    return u, fft_history, time_history
end

function exponential_filter(N; alpha=36.0, p=16)
    k = fftfreq(N, N)
    kmax = N ÷ 2
    sig = @. exp(-alpha * (abs(k) / kmax)^p)
    return sig
end

function cutoff_filter(N)
    k  = fftfreq(N, N)
    sig  = @. ifelse(abs(k) <= N ÷ 3, 1.0, 0.0)
    return sig
end

function apply_filter(u, sig)
    u_hat = fft(u)
    return real.(ifft(sig .* u_hat))
end
N = 33
x = (0:N-1) .* (2 * pi / N)
Tf = 3.0
dt = 0.002
eps = 0.01
a = 1
#initial condition
u0 = @. 0.5 + 0.25*sin(x)

Dx, Dxx = matrix_form(N)

L_inv = (N, u, a) -> L_inviscid(u, Dx)
L_vis = (N, u, a) -> L_viscous(u, Dx, Dxx, eps)

sig_exp    = exponential_filter(N; alpha=12, p=20)
sig_cutoff = cutoff_filter(N)

# 1. Inviscid runs
u_inv, fft_inv, _ = time_forward(u0, L_inv, Tf, dt, N, a, ssprk33)
u_inv_exp, fft_inv_exp, _ = time_forward(u0, L_inv, Tf, dt, N, a, ssprk33; filter=sig_exp)
u_inv_cut, fft_inv_cut, _ = time_forward(u0, L_inv, Tf, dt, N, a, ssprk33; filter=sig_cutoff)

# 2. Viscous runs (FIXED: Changed L_inv to L_vis)
u_vis, fft_vis, _ = time_forward(u0, L_vis, Tf, dt, N, a, ssprk33)
u_vis_exp, fft_vis_exp, _ = time_forward(u0, L_vis, Tf, dt, N, a, ssprk33; filter=sig_exp)
u_vis_cut, fft_vis_cut, _ = time_forward(u0, L_vis, Tf, dt, N, a, ssprk33; filter=sig_cutoff)

p1 = plot(x, u0, label="Initial", lw=2, legend=:bottom)
plot!(p1, x, u_inv, label="Inviscid (T=$(Tf))", lw=3, ls=:dot,color=:blue)
plot!(p1, x, u_vis, label="Viscous ε=$(eps)", lw=3, ls=:dot,color=:red)
plot!(p1, x, u_inv_exp, label="Inviscid + Exp filt",lw=2, ls=:solid,color=:purple)
plot!(p1, x, u_inv_cut, label="Inviscid + 2/3 filt",lw=2, ls=:solid, color=:orange)
plot!(p1, x, u_vis_exp, label="Viscous + Exp filt", lw=2, ls=:dash, color=:black)
plot!(p1, x, u_vis_cut, label="Viscous + 2/3 filt", lw=2, ls=:dash, color=:green)
xlabel!(p1, "x"); ylabel!(p1, "u"); title!(p1, "Burgers' Equation")
savefig(p1, "burger_eqn_nofilter.png")

k_axis = 0:N÷2
final_spectrum(fft_hist) =
    isempty(fft_hist) ? fill(NaN, N÷2+1) :
    log10.(fft_hist[end] .+ 1e-14)
 
p3 = plot(k_axis, final_spectrum(fft_inv), label="Inviscid", ls=:dot, lw=3)
plot!(p3, k_axis, final_spectrum(fft_vis), label="Viscous", ls=:dot, lw=3)
plot!(p3, k_axis, final_spectrum(fft_inv_exp), label="Inviscid+Exp", lw=2)
plot!(p3, k_axis, final_spectrum(fft_inv_cut), label="Inviscid+2/3", lw=2)
plot!(p3, k_axis, final_spectrum(fft_vis_exp), label="Viscous+Exp", lw=2)
plot!(p3, k_axis, final_spectrum(fft_vis_cut), label="Viscous+2/3", lw=2)
xlabel!(p3, "wavenumber k"); ylabel!(p3, "|u_k|  (log scale)")
title!(p3, "Fourier Spectrum at T=$(Tf)  (N=$(N))")
savefig(p3, "burger_fft_final.png")