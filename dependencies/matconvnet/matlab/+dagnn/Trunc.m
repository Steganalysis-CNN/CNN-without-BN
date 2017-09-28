classdef Trunc < dagnn.Filter
  properties
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nntrunc(inputs{1},[], obj.opts{:});
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nntrunc(inputs{1}, derOutputs{1}, obj.opts{:}) ;
      derParams = {} ;
    end
 end
end
