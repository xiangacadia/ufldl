function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units 
%                           (denoted in the lecture notes by the greek alphabet rho,
%                            which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  
% So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

%% Initilization value
% W1:25*64, W2:64*25, b1:25, b2:64
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions as b1, etc.  
% Your code should set W1grad to be the partial derivative of J_sparse(W,b) with respect to W1.  
% I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) with respect to the input parameter W1(i,j).  
% Thus, W1grad should be equal to the term [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] 
% in the last block of pseudo-code in Section 2.2 of the lecture notes 
% (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 

% cost = 0;
% W1grad = zeros(size(W1)); % W1:25*64
% W2grad = zeros(size(W2)); % W2:64*25
% b1grad = zeros(size(b1)); % b1:25
% b2grad = zeros(size(b2)); % b2:64

X = data; % the input layer:64*10000
[n_features, n_samples] = size(X);

% forward propagation
A{1} = X; % 64*10000
Z{2} = W1*A{1} + repmat(b1,1,n_samples); % 25*10000
A{2} = sigmoid(Z{2}); % 25*10000
Z{3} = W2*A{2} + repmat(b2,1,n_samples); % 64*10000
A{3} = sigmoid(Z{3}); % 64*10000

rho = sparsityParam;
% error back propagation
df_zL3 = A{3}.*(1-A{3}); % 64*10000 
delta_L3 = -(X-A{3}).*df_zL3; % 64*10000
% the hidden layer
df_zL2 = A{2}.*(1-A{2});% 25*10000
rho_2 = mean(A{2}, 2); % rho_2：  25*1，隐层单元的平均激活度
sparseterm = -rho./rho_2 + (1-rho)./(1-rho_2); 
% 隐层加入稀疏限制，会影响到隐层得到的delta_L2
delta_L2 = (W2'*delta_L3 + beta*repmat(sparseterm,1,n_samples)).*df_zL2; % 25*10000

W2grad = (1/n_samples)*delta_L3*A{2}' + lambda*W2; % 64*25
b2grad = (1/n_samples)*sum(delta_L3,2); % 64*1
W1grad = (1/n_samples)*delta_L2*A{1}' + lambda*W1; % 25*64
b1grad = (1/n_samples)*sum(delta_L2,2); % 25*1

cost = 0.5*(1/n_samples)*sum((X(:) - A{3}(:)).^2) + 0.5*lambda*sum(([W1(:);W2(:)]).^2) ...
     + beta*sum(rho.*log(rho./rho_2)+(1-rho).*log((1-rho)./(1-rho_2)));
 
%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

