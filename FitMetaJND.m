function [metaJND,C2deltaX,type2LapseRate,LL,exitflag,output] = FitMetaJND(deltaS,ConfCountRisLeft,ConfCountRisRight,PSE,Type1JND)
% [metaJND,C2deltaX,type2LapseRate,LL,exitflag,output] = FitMetaJND(deltaS,ConfCountRisLeft,ConfCountRisRight,PSE,Type1JND)
%
% Fit meta-JND based on the confidence responses from an observer at
% several different stimulus levels (deltaS). 
%
% INPUTS
%
% * deltaS
% One row vector (1 x nStimLevels) with the different stimulus levels
%
% * ConfCountRisLeft
% A matrix (nConfLevels x nStimLevels) with the number of trials on  
% which a participant responded with that confidence level (per stimulus
% level and given that the type-1 response R is "Left"). Different rows 
% represent different confidence levels in ascending order (i.e. the lowest 
% confidence level in row 1). The columns correspond to the stimulus levels 
% in "deltaS".
%
% * ConfCountRisRight
% The same matrix structure (nConfLevels x nStimLevels) as for 
% "ConfCountRisLeft", but these confidence level counts are of trials on 
% which the type-1 response R is "Right".
% 
% * PSE
% A scalar that determines the point of subjective equality from the type-1
% psychometric function. The type-1 PSE is necessary to compute an ideal 
% observer's theoretical probabilities for the confidence level responses.

% * Type1JND
% A scalar that determines the just noticeable difference from the type-1 
% psychometric function. The type-1 JND is used to provide an initial guess
% for the meta-JND parameter that will be subsequently optimized to best 
% fit the data. 
%
%
% OUTPUTs
%
% * metaJND
% A scalar that is the meta-JND parameter that provided the best fit to the
% confidence response data given stimulus levels "deltaS" and the "PSE".
%
% * C2deltaX
% One column vector with the nConfidenceLevels-1 type-2 criteria that form
% the borders of the different confidence level bins in units of deltaX.
%
% * type2LapseRate
% A scalar with the fitted type-2 lapse rate parameter. By default the
% lapse rate is bound between 0 and 0.25 (maximum).
%
% * LL
% A scalar that is the log-likelihood across all confidence responses
%
% * exitflag
% A scalar with the exitflag as returned by MATLAB's 'fmincon'
%
% * output
% A structure with the output structure as returned by MATLAB's 'fmincon'
%
%
% UPDATES
%
% 2018/07/19 - created by David Meijer


%Determine the number of parameters to fit
nConfLevels = size(ConfCountRisLeft,1);                                     %nStimLevels = size(ConfCountRisLeft,2);
nC2deltaX = nConfLevels-1;
nParams = nC2deltaX+2;                                                      %nC2deltaX + metaJND + type2LapseRate

%Set initial guesses for the parameters to fit
type2LapseRateGuess = 0.05;
C2deltaXFirstGuesses = linspace(0,max(deltaS),nConfLevels);
C2deltaXFirstGuesses = C2deltaXFirstGuesses(2:nConfLevels);                 %Delete first 0. That criterion is fixed.
params0 = [type2LapseRateGuess Type1JND C2deltaXFirstGuesses]';             %Parameter order: [type2LapseRate metaJND nC2deltaX]

% Set conditions for 'fmincon': A*x <= b
% Ensure A*x is negative
% Define A such that A*x reflects differences between C2deltaX: i.e. C2deltaX(K)-C2deltaX(K+1)    
nC2deltaXDiffs = nC2deltaX-1;
b = -eps*ones(nC2deltaXDiffs,1);
A = zeros(nC2deltaXDiffs,nParams);
for i=1:nC2deltaXDiffs
    A(i,i+2) = 1;
    A(i,i+3) = -1;
end

% Set lower and upper bounds for params
LB = [0; eps; eps*ones(nC2deltaX,1)];                                       %Parameter order: [type2LapseRate metaJND nC2deltaX]
UB = [0.25; inf; inf*ones(nC2deltaX,1)];

% Define fitting options
op = optimset(@fmincon);
op = optimset(op,'MaxFunEvals',100000,'Display','off');

%Call 'fmincon'
fitfun = @(params) ComputeNegLogL(params,deltaS,ConfCountRisLeft,ConfCountRisRight,PSE);
[params,negLL,exitflag,output] = fmincon(fitfun,params0,A,b,[],[],LB,UB,[],op);

%Collect parameters
type2LapseRate = params(1);
metaJND = params(2);
C2deltaX = params(3:end);
LL = -negLL;

end %[EOF]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Compute Negative Log Likelihood %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function negLogL = ComputeNegLogL(params,deltaS,ConfCountRisLeft,ConfCountRisRight,PSE)

%Compute the theoretical probabilities of all confidence levels at all stimulus levels for the given parameters    
[ConfProbRisLeft,ConfProbRisRight] = ComputeTheoreticProbs(params,deltaS,PSE);

%Sometimes the probability of a confidence level is zero. The log likelihood of that confidence level then is -inf, and the overall summed log-likelihood would also be -inf!   
%This is problematic since 'fmincon' can't estimate the gradient in the multidimensional log-likelihood landscape. Therefore, we ensure that no probability is ever zero.
ConfProbRisLeft = ConfProbRisLeft+eps;
ConfProbRisRight = ConfProbRisRight+eps;

%Compute the negative log likelihood across all responses    
logLikeLeft = sum(sum(ConfCountRisLeft.*log(ConfProbRisLeft)));             %There's no need for nansum here because the probability of a confidence level is never zero (see above)
logLikeRight = sum(sum(ConfCountRisRight.*log(ConfProbRisRight)));          %If that was not the case, then 0*log(0) would result in NaN. But this thus never happens (see above)
negLogL = -(logLikeLeft+logLikeRight);

end %[EOF "ComputeNegLogL"]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Compute Theoretic Probabilities of Confidence Levels %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ConfProbRisLeft,ConfProbRisRight] = ComputeTheoreticProbs(params,deltaS,PSE)

%Collect the parameters
type2LapseRate = params(1);
metaJND = params(2);
C2deltaX = params(3:end);

%Dimensions of the matrix
nStimLevels = numel(deltaS);
nConfLevels = numel(C2deltaX)+1;

StimLevelsMatrix = repmat(deltaS,[nConfLevels 1]);

%Create matrices with the criteria C2deltaX for the confidence level boundaries (domain: deltaX), separate for left and right responses    
C2deltaXRisRightK = repmat([0; C2deltaX],[1 nStimLevels]);
C2deltaXRisRightKplus1 = repmat([C2deltaX; inf],[1 nStimLevels]);

C2deltaXRisLeftK = repmat([0; -C2deltaX],[1 nStimLevels]);
C2deltaXRisLeftKplus1 = repmat([-C2deltaX; -inf],[1 nStimLevels]);

%The probability of a confidence level conditional on response z and StimLevel deltaS    
%I.e. The integral over the truncated Gaussian distribution P(deltaS_hat|z,deltaS) 
ConfProbRisRight = (erf((C2deltaXRisRightKplus1-StimLevelsMatrix+PSE)./(sqrt(2)*metaJND)) - erf((C2deltaXRisRightK-StimLevelsMatrix+PSE)./(sqrt(2)*metaJND))) ./ (1+erf((StimLevelsMatrix-PSE)./(sqrt(2)*metaJND)));
ConfProbRisLeft = (erf((C2deltaXRisLeftK-StimLevelsMatrix+PSE)./(sqrt(2)*metaJND)) - erf((C2deltaXRisLeftKplus1-StimLevelsMatrix+PSE)./(sqrt(2)*metaJND))) ./ erfc((StimLevelsMatrix-PSE)./(sqrt(2)*metaJND));    

%Sometimes the denominator of the above function can be zero (i.e. the probability of a response z (left or right) given deltaS is zero). 
%In that case the resulting probabilies for the confidence levels are NaN. This will cause a problem when computing the log-likelihood.      
%We correct for this by setting all NaNs to zero. 
ConfProbRisRight(isnan(ConfProbRisRight)) = 0;
ConfProbRisLeft(isnan(ConfProbRisLeft)) = 0;

%Add lapse rate support: some percentage of confidence responses might be given completely at random    
ConfProbRisRight = (1-type2LapseRate)*ConfProbRisRight + type2LapseRate*ones(nConfLevels,nStimLevels)./nConfLevels;
ConfProbRisLeft = (1-type2LapseRate)*ConfProbRisLeft + type2LapseRate*ones(nConfLevels,nStimLevels)./nConfLevels;

end %[EOF]
