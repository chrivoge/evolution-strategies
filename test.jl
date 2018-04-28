using InvPendulum, Plots

#Cost function
function cost_test(X)
return 1/(X[1]^2+X[2]^2);
end

function guete_test(theta,x,dt,T,costs)
    t = 0:dt:T;
    u = 0;
    X = zeros(2);
    X = x;
    J = 0;
    r = 0;
    for i=1:length(t)
        r = eval_nn(theta,X);
        J += costs(r);
    end
return J
end

function evolution_test(theta, alpha1, sigma, x1, J, dt, T, n, returns, costs)
    theta_optim = deepcopy(theta);
    X = deepcopy(x1);
    theta_vec = arr2vec(theta);
    gain = zeros(n,1);
    F_e = zeros(length(theta_vec),n);
    i = 1;
    while i<n
        epsilon_i = randn!(zeros(length(theta_vec))); #noise vector #maybe try map(x->x+randn!(zeros(1)),ones(length(theta_vec))) (faster?)
        theta_temp1 = vec2arr(theta_vec + sigma.*epsilon_i, theta); #add noise vector to the initial weights
        gain[i] = returns(theta_temp1,x1,dt,T,costs)[1]; #compute costs
        F_e[:,i] = gain[i]*epsilon_i;

        theta_temp1 = vec2arr(theta_vec - sigma.*epsilon_i, theta); #do the same for -epsilon_i
        gain[i+1] = returns(theta_temp1,x1,dt,T,costs)[1];
        F_e[:,i+1] = gain[i+1]*epsilon_i;
        i += 2;
    end
    theta_optim = vec2arr(theta_vec + alpha1*(1/n*sigma)*sum(F_e,2),theta);
    theta = theta_optim;
    return (theta_optim,returns(theta_optim,x1,dt,T,costs)[1]);
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

#train the neural network iteratively
function train(theta,x1,sigma,alpha1,returns, costs; maxiter=1000, dt = 1, T = 100, n = 1000)
    X = zeros(4);
    X = x1;
    J = zeros(maxiter);
    z = evolution_test(theta,alpha1,sigma,x1,J,dt,T,n,returns,costs);
    goal = returns(z[1],x1,dt,T,costs)[1];
    J[1] = goal;
    u = 0;
    i = 2;
    while i <= maxiter && J[i-1]/goal < 10
        z = evolution_test(z[1],alpha1,sigma,X,0,dt,T,n,returns,costs);
        J[i] = returns(z[1],x1,dt,T,costs)[1];
        i += 1;

    end
return (z,J);
end

#Try net on an episode
function run_episode(theta,x1; dt = 1, T = 100, n = 1000)
    t = 0:dt:T;
    X = zeros(2,length(t)+1);
    X[:,1] = x1;
    for i=1:length(t)
    u = eval_nn(theta,X[:,1]);
    X[:,i+1] = u;
    end
return X
end

#evaluate neural net
myrelu(x::T) where {T<:Number} = log(one(T)+exp(x))
myrelu(x) = log(1.0+exp(x)) # Needed for automatic differentiation

function eval_nn(w, x)
    for i=1:2:length(w)-2
        x =  myrelu.(w[i] * x .+ w[i+1]);
        x[.!isfinite.(x)]= 999;
    end
    return w[end-1] * x .+ w[end]
end

layers_Nneurons = [2, 16, 16, 8, 2];
#const rng = MersenneTwister(1234);

nn_params = [randn(layers_Nneurons[2], layers_Nneurons[1]),
randn(layers_Nneurons[2]),
randn(layers_Nneurons[3], layers_Nneurons[2]),
randn(layers_Nneurons[3]),
randn(layers_Nneurons[4], layers_Nneurons[3]),
randn(layers_Nneurons[4]),
randn(layers_Nneurons[5], layers_Nneurons[4]),
randn(layers_Nneurons[5])];
