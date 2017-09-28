function varargout = vl_nnwconv(x, param1, param2, param3, dzdy, varargin)
%VL_NNWCONV the weight normalized convolutional layer
% param1: the convolutional matrix w
% param2: the scalar vector g
% param3: the bias vector b
epsilon = 1e-05;
opts.pad = 1;
opts.stride = 1;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

v_norm = sqrt( sum(sum(sum(param1.^2, 1), 2), 3) + epsilon );
rep_v_norm = repmat(v_norm, [size(param1,1) size(param1,2) size(param1,3) 1]);
w = param1./rep_v_norm;

if isempty(dzdy)
    varargout{1} = vl_nnconv(x,w,param3,'pad', opts.pad, 'stride', opts.stride);
else
     [derInputs, derParam1, derParam2] = vl_nnconv(...
        x, w, param3, dzdy, ...
        'pad', opts.pad, ...
        'stride', opts.stride) ;
    varargout{1} = derInputs; % dzdy
     
    product_dw_v = sum( sum( sum(param1.*derParam1, 1), 2), 3 );
    der_g = product_dw_v ./v_norm;


    rep_param2 = gpuArray(single( zeros(1,1,1,size(param2,1)) ) );
    for i = 1 : size(param2,1)
      rep_param2(:,:,:,i) = param2(i);
    end

    der_v = ( repmat(rep_param2, [size(param1,1) size(param1,2) size(param1,3) 1]).* derParam1 ) ./rep_v_norm ...
            - ( repmat(rep_param2, [size(param1,1) size(param1,2) size(param1,3) 1]).* ...
            repmat(der_g, [size(param1,1) size(param1,2) size(param1,3) 1]) .* param1)./(rep_v_norm.^2);
    
    
    varargout{2} = der_v;
    varargout{3} = reshape(der_g, [size(param1,4) 1]);
    varargout{4} = derParam2; 
end
