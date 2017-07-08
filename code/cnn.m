function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

function [theta, meta] = cnnInitParams(cnnConfig)
% Initialize parameters
%                            
% Parameters:
%  cnnConfig    - cnn configuration variable
%
% Returns:
%  theta      -  parameter vector
%  meta       -  meta param 
%       numTotalParams : total number of the parameters
%       numParams      : the number of the parameters each layer

function [convolvedFeatures, linTrans] = cnnConvolve(images, W, b, nonlineartype, con_matrix, shape)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images

function [cost, grad, preds] = cnnCost(theta, images, labels,cnnConfig, meta, pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.

function [pooledFeatures, weights] = cnnPool(poolDim, convolvedFeatures, pooltypes)
%cnnPool Pools the given convolved features

function [output, linTrans] = nonlinear(input,W,b,type,norm)
%sigmoid computes nonlinear transformation of the input

cc/layer/Test.m: checks that all layers are working fine


function [opttheta] = minFuncSGD(funObj,theta,data,labels,...
                        options)
% Runs stochastic gradient descent with momentum to optimize the
% parameters for the given objective.

Helper:
function [Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,...
                                 numFilters,poolDim,numClasses)
% Converts unrolled parameters for a single layer convolutional neural
% network followed by a softmax layer into structured weight
% tensors/matrices and corresponding biases
