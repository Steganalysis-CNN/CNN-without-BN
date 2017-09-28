function bias_init(obj, inputs)

if ~iscell(inputs), error('INPUTS is not a cell array.') ; end
if obj.computingDerivative && ~iscell(derOutputs), error('DEROUTPUTS is not a cell array.') ; end

% -------------------------------------------------------------------------
% A Forward pass
% -------------------------------------------------------------------------

% set the input values
v = obj.getVarIndex(inputs(1:2:end)) ;
if any(isnan(v))
  broken = find(isnan(v)) ;
  error('No variable of name ''%s'' could be found in the DAG.', inputs{2*broken(1)-1}) ;
end
[obj.vars(v).value] = deal(inputs{2:2:end}) ;

obj.numPendingVarRefs = [obj.vars.fanout] ;
for l = obj.executionOrder
  fprintf('%d %s\n', l, obj.layers(l).name);
  obj.layers(l).block.forwardAdvanced(obj.layers(l)) ;
  layer_name = obj.layers(l).name;
  if strcmp(layer_name(1:3),'con')
      index_param_b =  obj.getParamIndex([layer_name '_b'])
      index_var_b = obj.getVarIndex([layer_name])
      var_bias = obj.getVar(layer_name);
      for i = 1 : size(var_bias.value, 3)
          var_out = var_bias.value(:,:,i,:);
          mu_t = mean(var_out(:));
          obj.params( index_param_b).value(i) = -mu_t;
          obj.vars(index_var_b).value(:,:,i,:) = obj.vars(index_var_b).value(:,:,i,:) - mu_t;
      end
  end
end
