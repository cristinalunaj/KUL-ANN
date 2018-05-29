function MSE = mse_calc(pred, target)
    MSE = mean((pred-target).^2);
end