%% matlab_demo_05.m
%%
%% CSS 490/590, Introduction to Machine Learning, Winter 2012
%% Computing & Software Systems, University of Washington Bothell
%% written by J. Jeffry Howbert, 2012-01-25

clear all
load heart
rng( 1 );
nInst = size( dat, 1 );
rp = randperm( nInst );
nTrain = 200;
nTest = nInst - nTrain;

trainDat = dat( rp( 1 : nTrain ), : );
trainLabel = label( rp( 1 : nTrain ) );
testDat = dat( rp( nTrain + 1 : end ), : );
testLabel = label( rp( nTrain + 1 : end ) );

B = mnrfit( trainDat, trainLabel );

% output from mnrval has two columns of probabilities, one for each class
pred = mnrval( B, trainDat );

% convert probabilities in column 2 into class labels (1 and 2) by comparing to threshold
threshold = 0.5;
trainPred = ( pred( :, 2 ) > threshold ) + 1;   % have to add 1 to get class labels 1 and 2

pred = mnrval( B, testDat );
testPred = ( pred( :, 2 ) > threshold ) + 1;

nTrainCorrect = sum( trainLabel == trainPred );
nTestCorrect = sum( testLabel == testPred );
fprintf( 1, 'correct predictions on training set :   %d / %d,  %5.2f%%\n', ...
    nTrainCorrect, nTrain, 100 * nTrainCorrect / nTrain );
[ mat, order ] = confusionmat( trainLabel, trainPred );
disp( ' ' );
disp( order' );
disp( mat );
fprintf( 1, 'correct predictions on test set     :   %d / %d,  %5.2f%%\n', ...
    nTestCorrect, nTest, 100 * nTestCorrect / nTest );
[ mat, order ] = confusionmat( testLabel, testPred, 'order', order );
disp( ' ' );
disp( order' );
disp( mat );
