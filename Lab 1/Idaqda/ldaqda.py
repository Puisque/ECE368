import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here

    num_males = np.sum(y == 1)
    num_females = np.sum(y == 2)
    mu_male = np.sum(x * (y == 1).reshape(len(y), -1), 0) / num_males
    mu_female = np.sum(x * (y == 2).reshape(len(y), -1), 0) / num_females
    cov_male = np.matmul(((x-mu_male) * (y == 1).reshape(len(y), -1)).T, ((x-mu_male) * (y == 1).reshape(len(y), -1))) / num_males
    cov_female = np.matmul(((x-mu_female) * (y == 2).reshape(len(y), -1)).T, ((x-mu_female) * (y == 2).reshape(len(y), -1))) / num_females
    cov = (num_males * cov_male + num_females * cov_female) / len(y)

    # plots for LDA and QDA function
    # x-axis: height
    # y-axis: weight
    # ranges from [50,80]x[80,280] , blue = male    red = female
    # 1) plot data points
    x_indices = np.where(y == 1)
    y_indices = np.where(y == 2)
    plt.figure(1)
    plt.scatter((x[x_indices].reshape(num_males*2,1))[0:2*num_males:2], (x[x_indices].reshape(num_males*2,1))[1:2*num_males:2], color='b')
    plt.scatter((x[y_indices].reshape(num_females*2,1))[0:2*num_females:2], (x[y_indices].reshape(num_females*2,1))[1:2*num_females:2], color='r')
    plt.figure(2)
    plt.scatter((x[x_indices].reshape(num_males*2,1))[0:2*num_males:2], (x[x_indices].reshape(num_males*2,1))[1:2*num_males:2], color='b')
    plt.scatter((x[y_indices].reshape(num_females*2,1))[0:2*num_females:2], (x[y_indices].reshape(num_females*2,1))[1:2*num_females:2], color='r')
    # 2) plot contours of conditional Gaussian Distribution for each class
    xx = np.linspace(50, 80, 100)
    yy = np.linspace(80, 280, 100)
    X, Y = np.meshgrid(xx, yy)
    LDA_m = []
    LDA_f = []
    QDA_m = []
    QDA_f = []
    x_val = X[0].reshape(100, 1)
    for i in range(100):
        # create X and Y values used to generate Z values for meshgrid
        data = np.concatenate((x_val, Y[i].reshape(100, 1)), 1)
        LDA_m.append(util.density_Gaussian(mu_male, cov, data))
        LDA_f.append(util.density_Gaussian(mu_female, cov, data))
        QDA_m.append(util.density_Gaussian(mu_male, cov_male, data))
        QDA_f.append(util.density_Gaussian(mu_female, cov_female, data))
    plt.figure(1)
    plt.contour(X, Y, LDA_m, colors='b')
    plt.contour(X, Y, LDA_f, colors='r')
    plt.figure(2)
    plt.contour(X, Y, QDA_m, colors='b')
    plt.contour(X, Y, QDA_f, colors='r')

# 3) plot contours of decision boundary, axis labels and titles
    LDA_boundary = np.array(LDA_m) - np.array(LDA_f)
    QDA_boundary = np.array(QDA_m) - np.array(QDA_f)
    plt.figure(1)
    plt.contour(X, Y, LDA_boundary, [0], colours='k')
    plt.title('Linear Discriminant Analysis')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.legend(['Male', 'Female'])
    plt.savefig('lda.pdf')
    plt.figure(2)
    plt.contour(X, Y, QDA_boundary, [0], colours='k')
    plt.title('Quadratic Discriminant Analysis')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.legend(['Male', 'Female'])
    plt.savefig('qda.pdf')
    plt.show()

    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here

    # convert to column vectors
    mu_male = mu_male.reshape(2, 1)
    mu_female = mu_female.reshape(2, 1)

    # LDA
    # note that since each xn feature should be a column vector and x is provided as a (110, 2), where each feature is
    # a row, the transpose is already computed
    LDA_y_m = np.dot(x, np.dot(np.linalg.inv(cov), mu_male)) - 0.5*np.dot(mu_male.T, np.dot(np.linalg.inv(cov), mu_male))
    LDA_y_f = np.dot(x, np.dot(np.linalg.inv(cov), mu_female)) - 0.5*np.dot(mu_female.T, np.dot(np.linalg.inv(cov), mu_female))
    LDA_y_hat = (LDA_y_f > LDA_y_m) + 1
    mis_lda = np.sum(y.reshape(len(y), -1) != LDA_y_hat) / len(y)

    # QDA
    # using loop
    # mis_qda = 0
    # for j, i in enumerate(x):
    #     QDA_y_m = - 0.5 * np.dot(np.dot(i - mu_male.T, np.linalg.inv(cov_male)), (i - mu_male.T).T) - np.log(np.sqrt(np.linalg.det(cov_male)))
    #     QDA_y_f = - 0.5 * np.dot(np.dot(i - mu_female.T, np.linalg.inv(cov_female)), (i - mu_female.T).T) - np.log(np.sqrt(np.linalg.det(cov_female)))
    #     QDA_y_hat = (QDA_y_f > QDA_y_m) + 1
    #     mis_qda += y[j] != QDA_y_hat
    # mis_qda = mis_qda[0][0]
    # using vectorized calculations, where the diagonal is the solution set
    QDA_y_m = - 0.5 * np.dot(np.dot(x - mu_male.T, np.linalg.inv(cov_male)), (x - mu_male.T).T) - np.log(np.sqrt(np.linalg.det(cov_male)))
    QDA_y_f = - 0.5 * np.dot(np.dot(x - mu_female.T, np.linalg.inv(cov_female)), (x - mu_female.T).T) - np.log(np.sqrt(np.linalg.det(cov_female)))
    QDA_y_m = np.diagonal(QDA_y_m)
    QDA_y_f = np.diagonal(QDA_y_f)
    QDA_y_hat = (QDA_y_f > QDA_y_m) + 1
    mis_qda = np.sum(y != QDA_y_hat) / len(y)

    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    # print(mis_LDA,mis_QDA)
    
    
    

    
