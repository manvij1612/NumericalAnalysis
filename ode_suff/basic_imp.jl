using Plots

function euler_method(Nsteps, t, y, dt)
    #=
    Euler's method for y' = cos(t) + y(t) with y(0) = 2
    =#
    for i = 1:Nsteps
        f = cos(t[i]) + y[i]
        y[i+1] = y[i] + dt * f
    end
    return y
end

function taylor_method_2(Nsteps, t, y, dt)
    #=
    Taylor's second order expansion for y' = cos(t) + y(t) with y(0) = 2
    =#
    for i = 1:Nsteps
        ft = cos(t[i]) + y[i]
        ft_p = -sin(t[i]) + ft
        y[i+1] = y[i] + ft * dt + ft_p * (dt^2/2)
    end
    return y
end

function runga_kutta_2(Nsteps, t, y, dt)
    #=
    Second order Runga-Kutta method for y' = cos(t) + y(t) with y(0) = 2
    =#
    for i = 1:Nsteps
        k1 = cos(t[i]) + y[i]
        k2 = cos(t[i] + dt/2) + (y[i] + dt*k1/2)
        y[i+1] = y[i] + k2*dt
    end
    return y
end

function runga_kutta_4(Nsteps, t, y, dt)
    #=
    Fourth order Runga-Kutta for y' = cos(t) + y(t) with y(0) = 2
    =#
    for i = 1:Nsteps
        k1 = cos(t[i]) + y[i]
        k2 = cos(t[i] + dt/2) + (y[i] + dt*k1/2)
        k3 =cos(t[i] + dt/2) + (y[i] + dt*k2/2)
        k4 = cos(t[i] + dt) + (y[i] + dt* k3 )
        y[i+1] = y[i] + 1/6 * dt * (k1 + k2*2 + k3*2 +k4)
    end
    return y
end

# Initialize parameters
Tfinal = 2.0
Nsteps_list = [10, 20, 50, 100, 500, 1000]

dt_values = Float64[]
errors_euler = Float64[]
errors_taylor = Float64[]
errors_rk2 = Float64[]
errors_rk4 = Float64[]

for Nsteps in Nsteps_list

    t = range(0, Tfinal, length=Nsteps+1)
    dt = t[2] - t[1]

    # Initialize solution array
    y = zeros(Nsteps+1) 
    y[1] = 2.0  # Initial condition

    y_euler = euler_method(Nsteps, t, copy(y), dt)
    y_taylor = taylor_method_2( Nsteps, t, copy(y), dt)
    y_runga_2 = runga_kutta_2(Nsteps, t, copy(y), dt)
    y_runga_4 = runga_kutta_4(Nsteps, t, copy(y), dt)

    # Analytical solution  
    y_analytical = 0.5 * (5 * exp.(t) .- cos.(t) .+ sin.(t))

    # Error calculation
    error_euler = abs.(y_euler - y_analytical)
    error_taylor = abs.(y_taylor - y_analytical)
    error_runga_2 = abs.(y_runga_2 - y_analytical)
    error_runga_4 = abs.(y_runga_4 - y_analytical)

    E_euler = maximum(error_euler)
    E_taylor = maximum(error_taylor)
    E_runga_2= maximum(error_runga_2)
    E_runga_4 = maximum(error_runga_4)
    
    println("For Nsteps = $Nsteps")
    @show E_euler
    @show E_taylor
    @show E_runga_2
    @show E_runga_4

    push!(dt_values, dt)
    push!(errors_euler, E_euler)
    push!(errors_taylor, E_taylor)
    push!(errors_rk2, E_runga_2)
    push!(errors_rk4, E_runga_4)

end

# Plot 1: Error vs dt
p1 = plot(dt_values, errors_euler, 
          label="Euler", marker=:circle, 
          xscale=:log10, yscale=:log10,
          xlabel="Step size (dt)", ylabel="Maximum Error",
          title="Convergence Analysis (Log-Log Scale)",
          linewidth=2, markersize=6, legend=:bottomright)

plot!(p1, dt_values, errors_taylor, 
      label="Taylor-2", marker=:square, 
      linewidth=2, markersize=6)

plot!(p1, dt_values, errors_rk2, 
      label="Runge-Kutta 2", marker=:diamond, 
      linewidth=2, markersize=6)

plot!(p1, dt_values, errors_rk4, 
      label="Runge-Kutta 4", marker=:star, 
      linewidth=2, markersize=6)

dt_ref = dt_values[2:end]
plot!(p1, dt_ref, 2.0*dt_ref.^1, label="O(dt)", linestyle=:dash, color=:black, linewidth=1.5)
plot!(p1, dt_ref, 0.5*dt_ref.^2, label="O(dt²)", linestyle=:dash, color=:gray, linewidth=1.5)
plot!(p1, dt_ref, 0.002*dt_ref.^4, label="O(dt⁴)", linestyle=:dash, color=:lightgray, linewidth=1.5)

display(p1)
savefig(p1, "convergence_plot.png")

# Plot 2: Error vs Nsteps
p2 = plot(Nsteps_list, errors_euler, 
          label="Euler", marker=:circle, 
          xlabel="Number of Steps", ylabel="Maximum Error",
          title="Error vs Number of Steps",
          linewidth=2, markersize=6, legend=:topright, yscale=:log10)

plot!(p2, Nsteps_list, errors_taylor, 
      label="Taylor-2", marker=:square, linewidth=2, markersize=6)

plot!(p2, Nsteps_list, errors_rk2, 
      label="Runge-Kutta 2", marker=:diamond, linewidth=2, markersize=6)

plot!(p2, Nsteps_list, errors_rk4, 
      label="Runge-Kutta 4", marker=:star, linewidth=2, markersize=6)

display(p2)
savefig(p2, "error_vs_steps.png")

