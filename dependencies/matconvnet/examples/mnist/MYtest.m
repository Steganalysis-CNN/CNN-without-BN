function [ output_args ] = MYtest3( input_args )
%对手写数字mnist训练集训练出来的模型进行测试,60000个训练，10000个测试。28*28*1


% 导入全体数据
load('E:\matlabCode\matconvnet-1.0-beta23\matconvnet-1.0-beta23\data\mnist-baseline-simplenn\imdb.mat');
% 挑选出测试样本在全体数据中对应的编号60001-70000
test_index = find(images.set==3);%1对应训练集，3对应测试集，1有（1——60000）3有（60001——70000）
% 挑选出测试集以及真实类别
test_data = images.data(:,:,:,test_index);
test_label = images.labels(test_index);
%导入训练好的模型（第20代的效果最好，所以选择此分类器）
load('E:\matlabCode\matconvnet-1.0-beta23\matconvnet-1.0-beta23\data\mnist-baseline-simplenn\net-epoch-20.mat');

% 将最后一层改为 softmax （原始为softmaxloss，这是训练用）
net.layers{1, end}.type = 'softmax';
% 对每张测试图片进行分类
for i = 1:length(test_label)
    i
    im_ = test_data(:,:,:,i);
    im_ = im_ - images.data_mean;
    res = vl_simplenn(net, im_) ;
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores) ;
    pre(i) = best;
   
end

% 计算准确率
disp(test_label);
disp(pre);
accurcy = length(find(pre==test_label))/length(test_label);
disp(['accurcy = ',num2str(accurcy*100),'%']);

  
end

