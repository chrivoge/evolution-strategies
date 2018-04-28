using InvPendulum, Plots

#Guetefunktion
function guete(theta,x1,dt,T,costs)
    t = 0:dt:T;
    X = zeros(4);
    X = x1;
    u = 0;
    J = 0;
    for i=1:length(t)
        u = eval_nn(theta,X);
        inv_pendulum_rk4!(X, u, dt);
        J += costs(X,u);
    end
    return J;
end

#Kostenfunktion
function cost_pend(x, u)
    k = [0.1,  0.1,  0.1,  1];
    ks = 0.0;
    if abs.(x[2]) > 2
        return 999999;
    else return k[3].*x[3].^2 + k[4].*(2-2.*cos(x[4])) + k[2].*x[2].^2 + k[1].*x[1].^2 + ks.*u.^2;
    end
end

#evaluate neural net
myrelu(x::T) where {T<:Number} = log(one(T)+exp(x))
myrelu(x) = log(1.0+exp(x)) # Needed for automatic differentiation

function eval_nn(w, x)
    for i=1:2:length(w)-2
        x =  myrelu.(w[i] * x .+ w[i+1]);
        x[.!isfinite.(x)]= 999;
    end
    return 10*tanh.((w[end-1] * x .+ w[end]))
end

#transform weight array to weight vector
function arr2vec(theta)
    theta_lin = zeros(0);
    temp = zeros(0);
    for i=1:length(theta)
        if length(size(theta[i])) == 2
            temp = reshape(theta[i],(size(theta[i],1)*size(theta[i],2), 1));
            theta_lin = [theta_lin; temp];
        else
            theta_lin = [theta_lin; theta[i]];
        end
    end
    return theta_lin;
end

#transform weight vector back to weight array
function vec2arr(theta_vec, theta_orig)
    theta_arr = Array{Array{Float64,N} where N,1}(length(theta_orig));
    c = 1;
    for i=1:length(theta_orig)
        dims = size(theta_orig[i]);
        if length(dims) == 2
            temp = reshape(theta_vec[c:(c - 1 + length(theta_orig[i]))], dims);
            theta_arr[i] = temp;
            c += length(theta_orig[i]);
        else
            theta_arr[i] = theta_vec[c:(c - 1 + length(theta_orig[i]))];
            c += length(theta_orig[i]);
        end
    end
    return theta_arr;
end

#Evolution Strategies Algorithm
function evolution_strat(theta, alpha1, sigma, x1, J, dt, T, n, returns, costs)
    theta_optim = deepcopy(theta);
    X = deepcopy(x1);
    theta_vec = arr2vec(theta);
    gain = zeros(n,1);
    F_e = zeros(length(theta_vec),n);
    i = 1;
    while i<n
        epsilon_i = randn!(zeros(length(theta_vec))); #noise vector
        theta_temp1 = vec2arr(theta_vec + sigma.*epsilon_i, theta); #add noise vector to the initial weights
        gain[i] = 1./returns(theta_temp1,x1,dt,T,costs)[1]; #compute costs
        F_e[:,i] = gain[i]*epsilon_i;

        theta_temp1 = vec2arr(theta_vec - sigma.*epsilon_i, theta); #do the same for -epsilon_i
        gain[i+1] = 1./returns(theta_temp1,x1,dt,T,costs)[1];
        F_e[:,i+1] = gain[i+1]*epsilon_i;
        i += 2;
    end
    theta_optim = vec2arr(theta_vec + alpha1*(1/n*sigma).*sum(F_e,2),theta);
    return theta_optim;
end

function episode(theta,x1,sigma,alpha1,returns, costs; maxiter=1000, dt = 0.01, T = 15, n = 2000)
    X = zeros(4);
    X = x1;
    J = 0;
    z = evolution_strat(theta,alpha1,sigma,x1,J,dt,T,n,returns,costs);
    u = 0;
    i = 1;
    while i < maxiter #|| ((J./(J-costs(X,eval_nn(z,)))) > 0.99 && J./(J-costs(z,X)) < 1.01)
        z = evolution_strat(z,alpha1,sigma,X,J,dt,T,n,returns,costs);
        #J = returns(z,x1,dt,T,costs)[1];
        i += 1;
    end
return z;
end

function run(theta,x1,evaluate; dt = 0.01, T = 15, n = 2000)
    t = 0:dt:T;
    X = zeros(4,length(t)+1);
    X[:,1] = x1;
    for i=1:length(t)
    u = eval_nn(theta,X[:,1]);
    X[:,i+1] = evaluate(X[:,i], u, dt);
    end
return X
end


const layers_Nneurons = [4, 16, 16, 8, 1];
#const rng = MersenneTwister(1234);

const nn_params = [0.001.*randn(layers_Nneurons[2], layers_Nneurons[1]),
0.001.*randn(layers_Nneurons[2]),
0.001.*randn(layers_Nneurons[3], layers_Nneurons[2]),
0.001.*randn(layers_Nneurons[3]),
0.001.*randn(layers_Nneurons[4], layers_Nneurons[3]),
0.001.*randn(layers_Nneurons[4]),
0.001.*randn(layers_Nneurons[5], layers_Nneurons[4]),
0.001.*randn(layers_Nneurons[5])];
