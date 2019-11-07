/*
y_t = \gamma + (X_t-\bar{X})\phi + \delta I_t + \epsilon_t

\gamma \sim N(g,G)      N((8,8),I)
\phi \sim N(p,P)        N((0.5,0.5),0.25I)
\delta \sim N(d,D)      N((0,0),[[0.1,-0.05],[-0.05,0.1]])
V \sim IW(w,W)          IW(3,I)

*/

data{
    int<lower=0> N;
    matrix[2,2] X_centered[N];
    vector[2] y[N];
    vector[N] I;
}

parameters{
    vector[2] gamma;
    vector[2] phi;
    vector[2] delta;
    cov_matrix[2] V;
}

model{

    gamma ~ multi_normal([8., 8.]', [[1., 0.],[0., 1.]]);
    phi ~ multi_normal([0.5, 0.5]', [[0.25, 0.],[0., 0.25]]);
    delta ~ multi_normal([0., 0.]', [[0.1,-0.05],[-0.05,0.1]]);
    V ~ inv_wishart(3, [[1., 0.], [0., 1.]]);

    for (n in 1:N){
        y[n] ~ multi_normal(gamma + X_centered[n] * phi + delta * I[n], V);
    }
}