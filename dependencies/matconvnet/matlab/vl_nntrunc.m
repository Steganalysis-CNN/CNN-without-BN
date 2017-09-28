function y = vl_nntrunc(x, dzdy, varargin)
%VL_NNTRUNC CNN rectified linear unit.
%   Y = VL_NNRELU(X) applies the rectified linear unit to the data
%   X. X can have arbitrary size.
%
%   DZDX = VL_NNRELU(X, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.
%
%   VL_NNRELU(...,'OPT',VALUE,...) takes the following options:
%
%   `Leak`:: 0
%      Set the leak factor, a non-negative number. Y is equal to X if
%      X is not smaller than zero; otherwise, Y is equal to X
%      multipied by the leak factor. By default, the leak factor is
%      zero; for values greater than that one obtains the leaky ReLU
%      unit.
%
%   ADVANCED USAGE
%
%   As a further optimization, in the backward computation it is
%   possible to replace X with Y, namely, if Y = VL_NNRELU(X), then
%   VL_NNRELU(X,DZDY) gives the same result as VL_NNRELU(Y,DZDY).
%   This is useful because it means that the buffer X does not need to
%   be remembered in the backward pass.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%v_x = x(:) .* (abs(x(:)) <= 5);
%i_x = ones(size(x(:))) .* (x(:) > 5) + (-1)*ones(size(x(:))) .* (x(:) < -5) ;
%res = (1-exp(-2*v_x))./(1+exp(-2*v_x)).*(v_x ~=0) + i_x .* (i_x ~= 0);
%res = reshape(res, size(x));

T = 5;
if nargin <= 1 || isempty(dzdy)
    y = x .* (abs(x) < T ) + T * (abs(x) >= T);
else
    y = dzdy .* (abs(x) < T);
end