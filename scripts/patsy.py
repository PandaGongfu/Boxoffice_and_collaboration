from patsy import dmatrices, dmatrix, demo_data

data = demo_data("a", "b", "x1", "x2", "y", "z column")

outcome, predictors = dmatrices("y ~ x1 + x2", data)
betas = np.linalg.lstsq(predictors, outcome)[0].ravel()