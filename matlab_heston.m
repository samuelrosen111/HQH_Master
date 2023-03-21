% MATLAB script to calculate the price of a call option using the Heston model

% Import Financial Instruments Toolbox functions
% First install finsymbols in python:"!pip install finsymbols"
%import finsymbols.*;
%import finpricer.*;
finsymbols = py.importlib.import_module('finsymbols');
symbols = finsymbols.get_sp500_symbols();


% Parameter inputs
S0 = 100.0;         % Initial asset price
v0 = 0.05;          % Initial variance
rho = -0.5;         % Correlation between asset returns and variance
kappa = 2.0;        % Rate of mean reversion in variance process
theta = 0.05;       % Long-term mean of variance process
sigma = 1.0;        % Volatility of volatility
T = 1.0;            % Time to maturity of the option (in years)
num_steps = 100;    % Number of time steps to simulate
num_sims = 10000;   % Number of Monte Carlo simulations to run
r = 0.02;           % Risk-free interest rate
option_type = 'call'; % Type of option to price
K = 110.0;          % Strike price of the option

% Create the Heston model object
hestonModel = finmodel.Heston('AssetPrice', S0, 'V0', v0, 'rho', rho, 'Kappa', kappa, ...
                               'Theta', theta, 'Sigma', sigma, 'NumSteps', num_steps, ...
                               'NumSims', num_sims);

% Create an option object
option = OptSpec(option_type, 'Strike', K, 'ExerciseStyle', 'european', 'OptionType', 'call');

% Create a rate specification
rateSpec = intenvset('ValuationDate', datetime(), 'StartDates', datetime(), 'EndDates', datetime() + calyears(T), 'Rates', r);

% Price the option using the Heston model
price = optPriceByHeston(hestonModel, option, rateSpec, T);

% Display the option price
fprintf('The price of the %s option with a strike of %0.2f and maturity of %0.2f years is %0.4f\n', option_type, K, T, price);
