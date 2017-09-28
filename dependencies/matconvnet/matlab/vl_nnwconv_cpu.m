function varargout = vl_nnwconv(x, param1, param2, param3, dzdy, varargin)
%VL_NNWCONV the weight normalized convolutional layer
% param1: the convolutional matrix w
% param2: the scalar vector g
% param3: the bias vector b

epsilon = 1e-05;
opts.pad = 1;
opts.stride = 1;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

v_norm = single(zeros(size(param1, 4),1));
w = single(zeros(size(param1)));
for i = 1 : size(param1, 4)
    param_i = param1(:,:,:,i);
    v_norm(i) = single(sqrt(norm(param_i(:)) + epsilon));
    w(:,:,:,i) = single( param1(:,:,:,i)*param2(i)/v_norm(i) );
end

if isempty(dzdy)
    varargout{1} = vl_nnconv(x,w,param3,'pad', opts.pad, 'stride', opts.stride);
else
     [derInputs, derParam1, derParam2] = vl_nnconv(...
        x, w, param3, dzdy, ...
        'pad', opts.pad, ...
        'stride', opts.stride) ;
    varargout{1} = derInputs; % dzdy
    
    der_g = single(zeros(size(param1,4),1));
    for i = 1 : size(param1,4)
        der_wi = derParam1(:,:,:,i);
        vi = param1(:,:,:,i);
        der_g(i) = single(  sum(vi(:).*der_wi(:))/v_norm(i) );
    end
    
    der_v = single(zeros(size(param1)));
    for i = 1  : size(param1, 4)
        der_v(:,:,:,i) = single( param2(i)*derParam1(:,:,:,i)/v_norm(i) ...
            - param2(i)*der_g(i)*param1(:,:,:,i)/(v_norm(i)*v_norm(i)) );
    end
    
    varargout{2} = der_v; % derivative of v
    varargout{3} = der_g; % derivative of g
    varargout{4} = derParam2;  % derivative of b
end
