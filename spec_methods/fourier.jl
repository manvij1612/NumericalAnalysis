using FFTW
using Plots
using LinearAlgebra
using Statistics

function L_operator(N, u, a)
    #=
    u(t) --> 1. FFT(u) --> 2. \hat{u_x} = ik \hat{u}_k --> 3. u_x = iFFT(\hat{u_x})
    L(u) = -a u_x
    =#
    k_wave = fftfreq(N,N) 
    u_hat = fft(u) #1
    du_hat = im .* k_wave .* u_hat #2
    ux = real(ifft(du_hat)) #3
    return -a .* ux
end

#Time stepping methods

function forward_euler(N, u, a, dt)
    return u .+ dt .* L_operator(N, u, a)
end

function ssprk22(N, u, a, dt)
    u1 = u .+ dt .* L_operator(N, u, a)
    return 0.5 .* u .+ 0.5 .* (u1 .+ dt .* L_operator(N, u1, a))
end

function ssprk33(N, u, a, dt)
    #Shu-osher RK3
    u1 = u .+ dt .* L_operator(N, u, a)
    u2 = 0.75 .* u .+ 0.25 .* (u1 .+ dt .* L_operator(N, u1, a))
    return (1/3) .* u .+ (2/3) .* (u2 .+ dt .* L_operator(N, u2, a))
end

function rk4(N, u, a, dt)
    k1 = L_operator(N, u, a)
    k2 = L_operator(N, u .+ 0.5 .* dt .* k1, a)
    k3 = L_operator(N, u .+ 0.5 .* dt .* k2, a)
    k4 = L_operator(N, u .+ dt .* k3, a)
    return u .+ (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

#marching forward in time
function time_forward(uinit, T, dt, N, a, evol_method)
    u = copy(uinit)
    nsteps = round(Int, T/dt)
    dt_real = T / nsteps
    for _ in 1:nsteps
        u = evol_method(N, u, a, dt_real)
    end
    return u
end

function exact_soln(uinit_func, x, a, t)
    return uinit_func.(mod.(x .- a .* t, 2 * pi))
end

function make_plots(N, uinit_func, a, t, dts, label)
    x = (0:N-1) .* (2 * pi/N) # x E [0, 2pi)
    uinit = uinit_func.(x)
    u_exact = exact_soln(uinit_func, x, a, t)

    methods = [("Forward Euler", forward_euler,  :red, :dash, :circle),
    ("SSPRK(2,2)", ssprk22, :blue, :dot, :square),
    ("SSPRK(3,3)", ssprk33, :green, :solid, :diamond),
    ("RK4", rk4, :purple, :dashdot, :star5)]

    #error vs dt plot
    p_err = plot(title="Error vs dt — $label (N=$N, T=$t)",
    xlabel="dt", ylabel="error",
    xscale=:log10, yscale=:log10,
    legend=:topleft, size=(700,500))

    #error vs exact solution plot
    p_exact = plot(title="Error vs exact solution — $label (N=$N, T=$t)",
    xlabel="exact solution", ylabel="error",
    legend=:topleft, size=(700,500))

    for (name, method, col, ls, mk) in methods
        errs = Float64[]
        valid_dts = Float64[]
        for dt in dts
            u_num = time_forward(uinit, t, dt,  N, a, method)
            err = maximum(abs.(u_num .- u_exact))
            if isfinite(err) && err < 1e4
                push!(errs, err)
                push!(valid_dts, dt)
            end
        end
        if length(valid_dts)>=2
            plot!(p_err, valid_dts, errs, label=name, lw=2, marker=mk, markersize=5, color=col)
            log_dt  = log10.(valid_dts)
            log_err = log10.(errs)
            order = (log_err[end] - log_err[1]) / (log_dt[end] - log_dt[1])
            println("$name  ($label): estimated order is $(round(order, digits=2))")
        end
    end
    return p_err
end

function matrix_form(N)
    Iden = Matrix(I, N, N)
    k = Diagonal(fftfreq(N,N))
    Dx   = ifft(ifftshift(im .* k * fftshift(fft(Iden, 1), 1), 1), 1)
    return real.(Dx)
end

a = 1.0
T = 2 * pi
N = 256

#defining initial conditions

uinit_smooth = x -> exp(sin(x))
uinit_nonsmooth = x -> sign(sin(x)) #square wave 

#varying dt
dts = 10.0 .^ range(-3, -0.5, length=15)

p_smooth = make_plots(N, uinit_smooth, a, T, dts, "Smooth")
p_nonsmooth = make_plots(N, uinit_nonsmooth, a, T, dts, "Non-Smooth")
savefig(p_smooth, "error_vs_t_smooth.png")
savefig(p_nonsmooth, "error_vs_t_nonsmooth.png")

#fourier differentiation matrix form

N_mat = 64
Dx = matrix_form(N_mat)
eigs = eigvals(-a .* Dx)

p_eig = scatter(real.(eigs), imag.(eigs),
    marker=:circle, markersize=4,
    xlabel="Re(eigen values)", ylabel="Im(eigen values)",
    title="Eigenvalues of L = -a·Dx  (N=$N_mat)", legend=true,
    aspect_ratio=:equal)
vline!([0.0], lw=1, ls=:dash, color=:black, label="")
savefig(p_eig, "eigenvalues_Dx.png")




