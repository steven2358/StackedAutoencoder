function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

% feedforward pass over autoencoders
depth = numel(stack);
z = cell(depth+1,1);
a = cell(depth+1,1);
a{1} = data;
for l=1:depth,
    wa = stack{l}.w*a{l};
    z{l+1} = bsxfun(@plus, wa, stack{l}.b);
    a{l+1} = sigmoid(z{l+1});
end

% feedforward over softmax
M = softmaxTheta*a{l+1};
M = bsxfun(@minus, M, max(M, [], 1)); % prevent overflow
h = exp(M); % hypotheses
h = bsxfun(@rdivide, h, sum(h)); % normalize

[mm, pred] = max(h, [], 1); %#ok<ASGLU>

% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
