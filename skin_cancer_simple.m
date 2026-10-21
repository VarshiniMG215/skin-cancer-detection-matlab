

clc; clear; close all; rng(42);

dataFolder = '/MATLAB Drive/archive';          % <-- folder with images + CSV
modelFile  = fullfile(dataFolder, 'skin_model.mat');
inputSize  = [227 227 3];
maxEpochs  = 5;

if ~exist(dataFolder, 'dir'), error('Folder not found: %s', dataFolder); end

%% 1. LOAD IMAGES AND LABELS ------------------------------------------------
imds = imageDatastore(dataFolder, 'IncludeSubfolders', true, ...
    'FileExtensions', {'.jpg', '.jpeg'});
[~, nm0, ~] = cellfun(@fileparts, imds.Files, 'UniformOutput', false);
[~, keep] = unique(string(nm0), 'stable');        % remove duplicate copies
imds = subset(imds, keep);

csvList = dir(fullfile(dataFolder, '**', '*metadata*.csv'));
if isempty(csvList), error('HAM10000_metadata.csv not found in %s', dataFolder); end
T = readtable(fullfile(csvList(1).folder, csvList(1).name));

[~, names, ~] = cellfun(@fileparts, imds.Files, 'UniformOutput', false);
[found, loc] = ismember(string(names), string(T.image_id));
imds = subset(imds, find(found));
rows = loc(found);
dx   = string(T.dx(rows));
lab  = repmat("benign", numel(dx), 1);
lab(ismember(dx, ["mel" "bcc" "akiec"])) = "malignant";
imds.Labels = categorical(lab);
lesionId = string(T.lesion_id(rows));

classNames = categories(imds.Labels);
fprintf('Images used: %d\n', numel(imds.Files));
disp(countEachLabel(imds));

%% 2. SPLIT: 70% train / 15% validation / 15% test (by lesion) --------------
[uL, ia] = unique(lesionId);
lesionLabel = imds.Labels(ia);
lesionSet = strings(numel(uL), 1);
for c = 1:numel(classNames)
    idx = find(lesionLabel == classNames{c});
    idx = idx(randperm(numel(idx)));
    n = numel(idx); nTr = round(0.70*n); nVa = round(0.15*n);
    lesionSet(idx(1:nTr)) = "train";
    lesionSet(idx(nTr+1:nTr+nVa)) = "val";
    lesionSet(idx(nTr+nVa+1:end)) = "test";
end
[~, li] = ismember(lesionId, uL);
imgSet = lesionSet(li);
imdsTrain = subset(imds, find(imgSet == "train"));
imdsVal   = subset(imds, find(imgSet == "val"));
imdsTest  = subset(imds, find(imgSet == "test"));

aug = imageDataAugmenter('RandXReflection', true, 'RandYReflection', true, ...
    'RandRotation', [-20 20]);
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', aug, 'ColorPreprocessing', 'gray2rgb');
augVal  = augmentedImageDatastore(inputSize(1:2), imdsVal,  'ColorPreprocessing', 'gray2rgb');
augTest = augmentedImageDatastore(inputSize(1:2), imdsTest, 'ColorPreprocessing', 'gray2rgb');

%% 3. TRAIN (skipped if a saved model already exists) -----------------------
if exist(modelFile, 'file')
    load(modelFile, 'net');
    disp('Saved model found - skipping training. (Delete skin_model.mat to retrain.)');
else
    lgraph = layerGraph(squeezenet);
    lgraph = replaceLayer(lgraph, 'conv10', convolution2dLayer(1, numel(classNames), ...
        'Name', 'new_conv', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10));
    counts = countcats(imdsTrain.Labels);
    w = sum(counts) ./ (numel(classNames) * counts);
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', ...
        classificationLayer('Name', 'out', 'Classes', classNames, 'ClassWeights', w));

    opts = trainingOptions('sgdm', 'MiniBatchSize', 32, 'MaxEpochs', maxEpochs, ...
        'InitialLearnRate', 3e-4, 'Shuffle', 'every-epoch', ...
        'ValidationData', augVal, ...
        'ValidationFrequency', floor(numel(imdsTrain.Files)/32), ...
        'Plots', 'none', 'Verbose', true, 'VerboseFrequency', 50);

    disp('Training started... (keep this tab open, takes about 20-40 minutes)');
    net = trainNetwork(augTrain, lgraph, opts);
    save(modelFile, 'net');
    disp('Training finished. Model saved.');
end

%% 4. ACCURACY ON UNSEEN TEST IMAGES ----------------------------------------
YPred = classify(net, augTest);
YTrue = imdsTest.Labels;
accuracy = mean(YPred == YTrue) * 100;
isM = YTrue == "malignant";  pM = YPred == "malignant";
sens = sum(isM & pM) / sum(isM) * 100;
spec = sum(~isM & ~pM) / sum(~isM) * 100;

disp('==============================================');
fprintf('  MODEL ACCURACY (test set) : %.2f %%\n', accuracy);
fprintf('  Sensitivity (cancer found): %.2f %%\n', sens);
fprintf('  Specificity (benign ok)   : %.2f %%\n', spec);
fprintf('  Test images               : %d\n', numel(YTrue));
disp('==============================================');

%% 5. PREDICT ANY IMAGE ------------------------------------------------------
classes = net.Layers(end).Classes;
mIdx = find(string(classes) == "malignant");
again = true;
while again
    [f, p] = uigetfile({'*.jpg;*.jpeg;*.png', 'Images'}, 'Select a skin image', dataFolder);
    if isequal(f, 0), break; end
    img = imread(fullfile(p, f));
    if size(img, 3) == 1, img = repmat(img, [1 1 3]); end
    img = imresize(img(:, :, 1:3), inputSize(1:2));
    [~, sc] = classify(net, img);
    pm = sc(mIdx) * 100;

    if pm >= 70,     level = 'HIGH chance of CANCER (malignant)';
    elseif pm >= 40, level = 'UNCERTAIN - could be cancer';
    else,            level = 'LOW chance of cancer (likely benign)';
    end

    fprintf('\nImage: %s\n', f);
    fprintf('Malignant (cancer) probability : %.1f %%\n', pm);
    fprintf('Benign probability             : %.1f %%\n', 100 - pm);
    fprintf('RESULT: %s\n', level);
    disp('Note: the model cannot measure how far cancer has spread. See a doctor.');

    figure('Name', 'Prediction'); imshow(img);
    title(sprintf('%s  |  Cancer probability: %.1f%%', level, pm), 'Interpreter', 'none');

    a = questdlg('Test another image?', 'Continue', 'Yes', 'No', 'Yes');
    again = strcmp(a, 'Yes');
end
disp('Done. Educational project - NOT a medical diagnosis.');
