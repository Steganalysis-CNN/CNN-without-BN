function imdb= cnn_steganalysis_setup_data(cover_path, stego_path, index)
% This function is to determine the training samples and testing samples

imdb.coverDir = cover_path;
imdb.stegoDir = stego_path;


% descriptions to the image database
imdb.meta.sets = {'train', 'val'};
imdb.meta.classes = {1,2};

% '1' represents the training sample, '2' represent the testing sample
set = [ones(1,5000) 2*ones(1,5000)];
imdb.images.set = set(index);

end