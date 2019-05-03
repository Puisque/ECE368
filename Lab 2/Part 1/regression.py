import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the prior distribution
    
    Outputs: None
    -----
    """
    # 1-D Array
    mean_vec = np.array([0, 0])
    # 2-D Array
    covariance_mat = np.array([[beta, 0], [0, beta]])

    # plot contour for prior distribution
    xx = np.linspace(-1, 1, 100)
    yy = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(xx, yy)
    x_val = X[0].reshape(100, 1)
    plot = []
    for i in range(100):
        # create X and Y values used to generate Z values for meshgrid
        x_set = np.concatenate((x_val, Y[i].reshape(100, 1)), 1)
        plot.append(util.density_Gaussian(mean_vec, covariance_mat, x_set))
    plt.figure(1)
    plt.title('Prior Distribution', fontsize=14)
    plt.xlabel(r'$a_{0}$', fontsize=12)
    plt.ylabel(r'$a_{1}$', fontsize=12)
    # plot Prior Distribution
    plt.contour(X, Y, plot, colors='red')
    # plot the actual point for A
    plt.plot([-0.1], [-0.5], marker='o', markersize=6, color='blue')
    # save the plot
    plt.savefig('./plots/prior.png')
    plt.savefig('./plots/prior.pdf')
    # plt.show()
    plt.close()
    return 


def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    n = len(x)
    cov_a = np.array([[beta, 0], [0, beta]])
    A = np.append(np.ones(shape=(n, 1)), x, axis=1)
    cov_w = np.array([sigma2])
    # 2-D Array
    Cov = np.linalg.inv(np.linalg.inv(cov_a) + np.matmul(A.T, A) / cov_w)
    # 1-D Array
    mu = np.matmul(Cov, np.matmul(A.T, z) / cov_w).reshape(1, 2).squeeze()
    # plot contour of posterior distribution p(a|x,z)
    xx = np.linspace(-1, 1, 100)
    yy = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(xx, yy)
    x_val = X[0].reshape(100, 1)
    plot = []
    for i in range(100):
        # create X and Y values used to generate Z values for meshgrid
        data = np.concatenate((x_val, Y[i].reshape(100, 1)), 1)
        plot.append(util.density_Gaussian(mu, Cov, data))
    plt.figure(1)
    if n == 1:
        plt.title(r'Posterior Distribution:  $p(a|x_{1},z_{1})$', fontsize=14)
    if n == 5:
        plt.title(r'Posterior Distribution:  $p(a|x_{1},z_{1},...,x_{5},z_{5})$', fontsize=14)
    if n == 100:
        plt.title(r'Posterior Distribution:  $p(a|x_{1},z_{1},...,x_{100},z_{100})$', fontsize=14)
    plt.xlabel(r'$a_{0}$', fontsize=12)
    plt.ylabel(r'$a_{1}$', fontsize=12)
    plt.contour(X, Y, plot, colors='red')
    # plot the actual point for A
    plt.plot([-0.1], [-0.5], marker='o', markersize=6, color='blue')
    # save the plot
    if n == 1:
        plt.savefig('./plots/posterior1.png')
        plt.savefig('./plots/posterior1.pdf')
    if n == 5:
        plt.savefig('./plots/posterior5.png')
        plt.savefig('./plots/posterior5.pdf')
    if n == 100:
        plt.savefig('./plots/posterior100.png')
        plt.savefig('./plots/posterior100.pdf')
    # plt.show()
    plt.close()
    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    # create A matrix where column one corresponds to 'ones' and column two corresponds to 'x'
    n = len(x_train)
    A = np.append(np.ones([len(x), 1]), np.expand_dims(x, 1), axis=1)

    # compute the predictions, which is data*means
    z = np.matmul(A, mu)
    # Note that the mean of z is z itself (since z is gaussian distributed with centre, mu)
    MU_z = z

    # compute the covariance matrix for z, and take the diagonal to find the variances and thus stds of each mean
    cov_w = np.array([sigma2])
    var_z = cov_w + np.matmul(np.matmul(A, Cov), A.T)
    std_z = np.sqrt(np.diag(var_z))

    plt.figure(1)
    if n == 1:
        plt.title(r'Prediction Distribution:  $p(z|x,x_{1},z_{1})$', fontsize=14)
    if n == 5:
        plt.title(r'Prediction Distribution:  $p(z|x,x_{1},z_{1},...,x_{5},z_{5})$', fontsize=14)
    if n == 100:
        plt.title(r'Prediction Distribution:  $p(z|x,x_{1},z_{1},...,x_{100},z_{100})$', fontsize=14)
    plt.xlabel('input: x', fontsize=12)
    plt.ylabel('output: z', fontsize=12)
    # set range to be [−4, 4] × [−4, 4]
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    # (1) and (2)
    plt.errorbar(x, MU_z, yerr=std_z, fmt='rx', label='Testing Samples')
    # (3)
    plt.scatter(x_train, z_train, s=4, color='blue', label='Training Samples')
    plt.legend()
    # save the plot
    if n == 1:
        plt.savefig('./plots/predict1.png')
        plt.savefig('./plots/predict1.pdf')
    if n == 5:
        plt.savefig('./plots/predict5.png')
        plt.savefig('./plots/predict5.pdf')
    if n == 100:
        plt.savefig('./plots/predict100.png')
        plt.savefig('./plots/predict100.pdf')
    # plt.show()
    plt.close()
    return


if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    for ns in [1, 5, 100]:
        # used samples
        x = x_train[0:ns]
        z = z_train[0:ns]

        # prior distribution p(a)
        priorDistribution(beta)

        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)

        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
