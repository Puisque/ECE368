import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here

    # separate spam from ham and count number of emails in training set
    [spam, ham] = file_lists_by_category
    num_spam = len(spam)
    num_ham = len(ham)

    p = util.Counter()
    q = util.Counter()
    # get counts of each word in spam and ham
    counts_spam = util.get_counts(spam)
    counts_ham = util.get_counts(ham)
    # build vocab for spam and ham
    spam_vocab = list(counts_spam)
    ham_vocab = list(counts_ham)
    vocab = list(set(spam_vocab + ham_vocab))

    # estimate p and q for all d in vocab d = 1, ..., D using Laplace smoothing, k = 1, classes = 2
    k = 1
    classes = len(file_lists_by_category)
    for w in vocab:
        p[w] = (counts_spam[w] + k) / (num_spam + k * classes)
        q[w] = (counts_ham[w] + k) / (num_ham + k * classes)

    # concatenate into tuple
    probabilities_by_category = p, q

    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category,b):
    # added a bias term, b to modify decision rule for part 1.2 (c)
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    
    probabilities_by_category: output of function learn_distributions
    
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here

    # extract vocab, probability estimates and prior probabilities
    x = util.get_words_in_file(filename)
    p, q = probabilities_by_category
    prior_spam, prior_ham = prior_by_category

    # calculate value of email being spam
    log_spam = 0
    for w in p:
        x_d = int(w in x)
        # either implementation works
        # log_spam += np.log(p[w] ** x_d * (1 - p[w]) ** (1 - x_d))
        log_spam += x_d*np.log(p[w]) + (1 - x_d)*np.log((1 - p[w]))
    log_spam += np.log(prior_spam)

    # calculate value of email being ham
    log_ham = 0
    for w in q:
        x_d = int(w in x)
        # log_ham += np.log((q[w] ** x_d * (1 - q[w]) ** (1 - x_d)))
        log_ham += x_d*np.log(q[w]) + (1 - x_d)*np.log((1 - q[w]))
    log_ham += np.log(prior_ham)

    # compare and return the result
    if log_spam + b >= log_ham:
        label = 'spam'
        log_posterior = log_spam
    else:
        label = 'ham'
        log_posterior = log_ham
    classify_result = label, log_posterior

    return classify_result


if __name__ == '__main__':

    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))

    # section 1.1 (b)
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)

    # prior class distribution
    priors_by_category = [0.5, 0.5]

    # section 1.2 (b)
    ''''''
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham'

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        ### added below to determine difference between posteriors for part 1.2 (c)
        label,log_posterior = classify_new_email(filename,
                                                    probabilities_by_category,
                                                    priors_by_category,b=0)

        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    ''''''

    # section 1.2 (c)
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve

    # modify classify_new_email to have an extra parameter s.t. there is a bias in the function call that will be added
    # onto the decision rule, chosen from the differences between the original posterior differences
    ''''''
    type_I_error = []
    type_II_error = []
    # given that at a bias of -35, 0 Type II errors occur and at a 25, 0 Type I Errors occur, loop from -35 to 25, at
    # an increment of 5 (so 13 points, -35 and 25 inclusive)
    # biases = [-35, 25]
    b = -35
    while b <= 25:
        # print(b)
        performance_measures = np.zeros([2, 2])
        # Classify emails from testing set and measure the performance
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label, log_posterior = classify_new_email(filename,
                                                          probabilities_by_category,
                                                          priors_by_category, b)

            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base)
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0], totals[0], correct[1], totals[1]))
        type_I_error.append(totals[0]-correct[0])
        type_II_error.append(totals[1]-correct[1])
        b += 5
    np.savetxt('type_I_error.csv', type_I_error, delimiter=',')
    np.savetxt('type_II_error.csv', type_II_error, delimiter=',')
    # print(type_I_error)
    # print(type_II_error)
    ''''''

    # plot trade-off curve
    type_I_error = np.loadtxt('type_I_error.csv', delimiter=',')
    type_II_error = np.loadtxt('type_II_error.csv', delimiter=',')
    plt.plot(type_I_error, type_II_error)
    plt.title('Trade-off Curve for SPAM Filtering (Naive Bayes Classifier)')
    plt.ylabel('Type II Error')
    plt.xlabel('Type I Error')
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.savefig('nbc.pdf')
    plt.show()
