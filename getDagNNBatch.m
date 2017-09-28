function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
% step up parameters 
opts.imageSize = [256, 256] ;
opts.border = [0, 0] ; 
opts.keepAspect = true ;
opts.numAugments = 1 ;
opts.transformation = 'none'; 
opts.affine = false;
opts.averageImage = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;

 sample_path = strcat(imdb.coverDir, num2str(batch(1)), '.pgm');
 cover = imread(sample_path);
 im = single(zeros(size(cover,1),size(cover,2),1,2*length(batch)));
 
for i = 1 : length(batch)
   cover_path = strcat(imdb.coverDir, num2str(batch(i)), '.pgm'); 
   stego_path = strcat(imdb.stegoDir, num2str(batch(i)), '.pgm');
   cover = imread(cover_path);
   stego = imread(stego_path);
   
   im(:, :, 1, 2*i-1) = single(cover);
   im(:, :, 1, 2*i) = single(stego);
end
 
labels = ones(1,2*length(batch)) + (sign((-1).^(1:2*length(batch)))+1)/2; % set the label, 1: cover; 2: stego. 

if nargout > 0
    if useGpu
        im = gpuArray(im) ;
    end
    inputs = {'data', im, 'label', labels} ;
end

end