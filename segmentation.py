import numpy as np

def default_convergence(previous_loglikelihood, current_loglikelihood, convergence_counter, conv_ctr_cap=5,
                        loglikelihood_threshold=1e-6):
    """
    Default convergence function that checks if the log likelihood improvement
    is below a threshold.

    params:
    previous_loglikelihood = float
    current_loglikelihood = float
    convergence_counter = int
    conv_ctr_cap = int (optional, default=5)
    loglikelihood_threshold = float (optional, default=1e-6)

    returns:
    (convergence_counter, converged)
    """
    if abs(current_loglikelihood - previous_loglikelihood) < loglikelihood_threshold:
        convergence_counter += 1
    else:
        convergence_counter = 0

    converged = convergence_counter >= conv_ctr_cap
    return convergence_counter, converged



def get_initial_means(array, k):
    """
    Picks k random points from the 2D array
    (without replacement) to use as initial
    cluster means

    params:
    array = numpy.ndarray[numpy.ndarray[float]] - m x n | datapoints x features

    k = int

    returns:
    initial_means = numpy.ndarray[numpy.ndarray[float]]
    """
    clusters = np.random.randint(len(array),size=k)
    initial_cluster_means = array[clusters,:]
    return initial_cluster_means


def k_means_step(X, k, means):
    """
    A single update/step of the K-means algorithm
    Based on a input X and current mean estimate,
    predict clusters for each of the pixels and
    calculate new means.
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n | pixels x features (already flattened)
    k = int
    means = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    (new_means, clusters)
    new_means = numpy.ndarray[numpy.ndarray[float]] - k x n
    clusters = numpy.ndarray[int] - m sized vector
    """
    ''' get the distance of each point at the respective cluster '''
    distances = np.sqrt(((X - means[:, np.newaxis]) ** 2).sum(axis=2))
    ''' get the shortest distances indices '''
    distances = np.argmin(distances, axis=0)
    '''move cluster points to mean/cluster location'''
    new_means = np.array([X[distances == i].mean(axis=0) for i in range(means.shape[0])])

    return new_means, distances


def k_means_segment(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    """
    old_clusters = np.zeros(shape=len(image_values) * len(image_values[0]), dtype=int)
    new_clusters = np.zeros(shape=len(image_values) * len(image_values[0]), dtype=int)
    old_means = initial_means
    first_round = True

    if initial_means is None:
        ''' get initial means and reshape image values to be an array with 3 columns '''
        initial_means = get_initial_means(image_values.reshape(-1,3), k)
        old_means = initial_means[:]


    '''keep selecting cluseters and assigning all points on the graph to those clusters/means until the last iteration output an identical cluster'''
    while True:
        ''' if hew new cluster gathered is identical to the previous cluseter we gathered, then we're done'''
        if np.array_equal(old_clusters, new_clusters) and first_round is not True:
            '''reshape our new image by giving it back the color channel'''
            updated_image = np.reshape(new_means[new_clusters], (image_values.shape[0], image_values.shape[1], image_values.shape[2]))
            return updated_image
        else:
            old_clusters = new_clusters[:]
            new_means, new_clusters = k_means_step(image_values.reshape(-1,3), k, old_means)
            old_means= new_means[:]
            first_round = False


def compute_sigma(X, MU):
    """
    Calculate covariance matrix, based in given X and MU values

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    """
    k = MU.shape[0]
    SIGMA = (
        np.dot( np.transpose(X - MU[i]), (X - MU[i])) for i in range(k)
    )
    SIGMA = list(SIGMA)
    SIGMA = np.array(SIGMA, dtype=float)
    SIGMA *= 1/len(X)

    return SIGMA

def initialize_parameters(X, k):
    """
    Return initial values for training of the GMM
    Set component mean to a random
    pixel's value (without replacement),
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int

    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    """
    ''' get means'''
    MU = get_initial_means(X, k)

    '''get covariance matrix (basically standard deviation but for multivariate data)'''
    SIGMA = compute_sigma(X, MU)

    PI = np.ndarray((k,1), dtype=float)
    ''' fill PI with 1/k which pi == the probability of any existing datapoint being in either classification '''
    PI[:] = 1.0/k

    return MU, SIGMA, PI


def prob(x, mu, sigma):
    """Calculate the probability of x (a single
    data point or an array of data points) under the
    component with the given mean and covariance.
    The function is intended to compute multivariate
    normal distribution, which is given by N(x;MU,SIGMA).

    params:
    x = numpy.ndarray[float] (for single datapoint)
        or numpy.ndarray[numpy.ndarray[float]] (for array of datapoints)
    mu = numpy.ndarray[float]
    sigma = numpy.ndarray[numpy.ndarray[float]]

    returns:
    probability = float (for single datapoint)
                or numpy.ndarray[float] (for array of datapoints)
    """

    ''' single row of data'''
    if len(x.shape) == 1:
        n = len(x)
        '''constant from prob equation'''
        constant = 1 / ((2 * 3.1415926) ** (n/2) * np.linalg.det(sigma) ** 0.5)
        probability = constant * np.exp(-0.5 * ( np.transpose(x - mu) @ np.linalg.inv(sigma) @ (x - mu) ))
        return probability
    else:
        n = len(x[0])
        '''constant from prob equation'''
        constant = 1 / ((2 * 3.1415926) ** (n / 2) * np.linalg.det(sigma) ** 0.5)
        ''' axis 1 (x axis) == perform on rows, axis 0 (y axis) = perform on columns '''
        probability = constant * np.exp(-0.5 * np.sum((x - mu) @ np.linalg.inv(sigma) * (x - mu), axis=1))
        probability = np.array(probability)
        return probability

def E_step(X,MU,SIGMA,PI,k):
    """
    E-step - Expectation
    Calculate responsibility for each
    of the data points, for the given
    MU, SIGMA and PI.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int

    returns:
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    responsibility = np.ndarray(shape=(k, len(X)), dtype=float)
    n = len(X[0])
    ''' cannot call prob() because of dimensions getting in the way, need to write prob all over again'''
    for i in range(k):
        ''' do the prob() equation again but because we're dealing with X and not x, we gotta np.expand for the exponent part of the prob() equation '''
        constant = 1 / ((2 * 3.1415926) ** (n/2) * np.sqrt(np.linalg.det(SIGMA[i])))
        probability = constant * np.exp( -0.5*np.expand_dims(np.sum((X - MU[i] ) @ np.linalg.inv(SIGMA[i]) * (X - MU[i]), axis=1), axis=0) )
        prob_component = probability[0]
        prob_component = np.reshape(prob_component, (1, prob_component.shape[0]))
        responsibility[i] = np.dot(PI[i], prob_component)

    final_responsibility = np.divide(responsibility, np.sum(responsibility, axis=0))
    return final_responsibility


def M_step(X, r, k):
    """
    M-step - Maximization
    Calculate new MU, SIGMA and PI matrices
    based on the given responsibilities.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    r = numpy.ndarray[numpy.ndarray[float]] - k x m
    k = int

    returns:
    (new_MU, new_SIGMA, new_PI)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︉͏︆͏󠄃
    m = len(X)
    n = len(X[0])

    '''get new pi'''
    m_components = np.sum(r, axis=1)
    new_pi = np.divide(m_components, m)

    '''get new mu (matmul is matrix multiplication (@) and dot (*) is just element wise multiplication and allows scalars'''
    r_and_x = np.matmul(np.transpose(X), np.transpose(r))
    new_mu =  np.transpose(np.divide(r_and_x, m_components))
    new_mu = new_mu.reshape(k,n)

    ''' get new sigma'''
    new_sigma = np.ndarray(shape=(k, n, n), dtype=float)

    for i in range(k):
        ''' get first product via r[i] * X-MU[i] (reshape r into a vector form because python sucks)'''
        first_product = (X - new_mu[i]) * np.reshape(r[i], (len(r[i]), 1))
        new_sigma[i] = ( np.transpose(first_product) @ (X - new_mu[i])) * ( 1 / m_components[i] )

    return new_mu, new_sigma, new_pi

def loglikelihood(X, PI, MU, SIGMA, k):
    """Calculate a log likelihood of the
    trained model based on the following
    formula for posterior probability:

    log(Pr(X | mixing, mean, stdev)) = sum((i=1 to m), log(sum((j=1 to k),
                                      mixing_j * N(x_i | mean_j,stdev_j))))

    Make sure you are using natural log, instead of log base 2 or base 10.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int

    returns:
    log_likelihood = float
    """
    responsibility = np.ndarray(shape=(k, len(X)), dtype=float)
    n = len(X[0])

    ''' repeat of the E_Step'''
    i = 0
    while i < k:
        constant = 1 / ((2 * 3.1415926) ** (n/2) * np.sqrt(np.linalg.det(SIGMA[i])))
        probability = constant * np.exp( -0.5*np.expand_dims(np.sum((X - MU[i] ) @ np.linalg.inv(SIGMA[i]) * (X - MU[i]), axis=1), axis=0) )
        prob_component = probability[0]
        prob_component = np.reshape(prob_component, (1, prob_component.shape[0]))
        responsibility[i] = np.dot(PI[i], prob_component)
        i += 1

    ''' sum all responsibilities on axis 0, then log it , then sum it again '''
    log_likelihood = np.sum(np.log(np.sum(responsibility, axis=0)))
    return log_likelihood


def train_model(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the
    expectation-maximization algorithm.
    E.g., iterate E and M steps from
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True,
    see default convergence_function example
    in `helper_functions.py`

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    if initial_values is None:
        initial_values = initialize_parameters(X, k)

    ''' initial_values[0] = MU, initial_values[1] = SIGMA, initial_values[2] = PI'''
    previous_loglikelihood = 0.0
    convergence_counter = 0

    i = 0
    while i < 100:
        ''' get likelihood '''
        current_loglikelihood = loglikelihood(X, initial_values[2], initial_values[0], initial_values[1], k)
        ''' get expectation '''
        responsibility = E_step(X, initial_values[0], initial_values[1], initial_values[2], k)
        ''' get maximization '''
        initial_values = M_step(X, responsibility, k)
        convergence_counter, converge = convergence_function(previous_loglikelihood, current_loglikelihood, convergence_counter)

        ''' if they converge then return'''
        if converge:
            return initial_values[0], initial_values[1], initial_values[2], responsibility

        previous_loglikelihood = current_loglikelihood
        i += 1

    return initial_values[0], initial_values[1], initial_values[2], responsibility


def cluster(r):
    """
    Based on a given responsibilities matrix
    return an array of cluster indices.
    Assign each datapoint to a cluster based,
    on component with a max-likelihood
    (maximum responsibility value).

    params:
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix

    return:
    clusters = numpy.ndarray[int] - m x 1
    """
    r_clusters = np.argmax(r, axis=0)
    return r_clusters


def segment(X, MU, k, r):
    """
    Segment the X matrix into k components.
    Returns a matrix where each data point is
    replaced with its max-likelihood component mean.
    E.g., return the original matrix where each pixel's
    intensity replaced with its max-likelihood
    component mean. (the shape is still mxn, not
    original image size)

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    k = int
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix

    returns:
    new_X = numpy.ndarray[numpy.ndarray[float]] - m x n
    """
    segment = MU[cluster(r)]
    return segment


def best_segment(X,k,iters):
    """Determine the best segmentation
    of the image by repeatedly
    training the model and
    calculating its likelihood.
    Return the segment with the
    highest likelihood.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    iters = int

    returns:
    (likelihood, segment)
    likelihood = float
    segment = numpy.ndarray[numpy.ndarray[float]]
    """
    best_likelihood = 0.0
    best_results = []
    initial_values = initialize_parameters(X, k)
    training_results = initial_values

    i = 0
    ''' run the training '''

    while i < iters:
        ''' initial_values[0] = MU, initial_values[1] = SIGMA, and intial_values[2] = PI '''
        training_results = train_model(X, k, default_convergence, training_results)
        ''' get likelihood '''
        current_likelihood = loglikelihood(X, training_results[2], training_results[0], training_results[1], k)
        ''' update likelihood and the best results for the training '''
        if current_likelihood > best_likelihood:
            best_likelihood = current_likelihood
            best_results = training_results

        i += 1


    if len(best_results) == 0:
        training_results = train_model(X, k, default_convergence, training_results)
        better_X = segment(X, training_results[0], k, training_results[3])
        current_likelihood = loglikelihood(better_X, training_results[2], training_results[0], training_results[1], k)
        return current_likelihood, better_X

    better_X = segment(X, training_results[0], k, training_results[3])
    best_likelihood = loglikelihood(better_X, training_results[2], training_results[0], training_results[1], k)

    return best_likelihood, better_X


def improved_initialization(X,k):
    """
    Initialize the training
    process by setting each
    component mean using some algorithm that
    you think might give better means to start with,
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int

    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    """
    trained_model = train_model(X, k, default_convergence)
    return trained_model[0], trained_model[1], trained_model[2]


def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:
    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    (conv_crt, converged)
    conv_ctr = int
    converged = boolean
    """
    if previous_variables[0] is None or new_variables[0] is None:
        return 0, False

    ''' converge based on how close the means and variances are '''
    if np.allclose(previous_variables[0], new_variables[0], rtol=0.1) and np.allclose(previous_variables[1], new_variables[1], rtol=0.1) and np.allclose(previous_variables[2], new_variables[2], rtol=0.1):
        conv_ctr += 1
    else:
        conv_ctr = 0

    converge = conv_ctr > conv_ctr_cap
    return conv_ctr, converge



def train_model_improved(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the
    expectation-maximization algorithm.
    E.g., iterate E and M steps from
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True. Use new_convergence_fuction
    implemented above.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    convCounter = 0
    previous_params = np.empty(shape=3, dtype=object)

    if initial_values is None:
        current_params = improved_initialization(X, k)
    else:
        current_params = initial_values


    i = 0
    while i < 900:
        convCounter, converge = new_convergence_function(previous_params, current_params, convCounter)

        if converge:
            break

        previous_params = current_params
        responsibility = E_step(X, current_params[0], current_params[1], current_params[2], k)
        current_params = M_step(X, responsibility, k)
        i += 1

    return current_params[0], current_params[1], current_params[2], responsibility


def bayes_info_criterion(X, PI, MU, SIGMA, k):
    """
    See description above
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
     numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int

    return:
    bayes_info_criterion = int
    """
    m = X.shape[0]
    n = X.shape[1]

    Bayes = np.log(m) * (k*n + k*n * (n+1)/2+k ) -2 * loglikelihood(X, PI, MU, SIGMA, k)
    return Bayes


def BIC_likelihood_model_test(image_matrix, comp_means):
    """Returns the number of components
    corresponding to the minimum BIC
    and maximum likelihood with respect
    to image_matrix and comp_means.

    params:
    image_matrix = numpy.ndarray[numpy.ndarray[float]] - m x n
    comp_means = list(numpy.ndarray[numpy.ndarray[float]]) - list(k x n) (means for each value of k)

    returns:
    (n_comp_min_bic, n_comp_max_likelihood)
    n_comp_min_bic = int
    n_comp_max_likelihood = int
    """
    BIC = 100000.0
    LOG_LIKE = -100000.0

    for i in range(len(comp_means)):
        k = comp_means[i].shape[0]

        MU = comp_means[i]
        SIGMA = compute_sigma(image_matrix, MU)

        PI = np.ones(k)
        PI[:] = 1.0 / k

        '''train model'''
        MU, SIGMA, PI, responsibility = train_model_improved(image_matrix, k, default_convergence, [MU, SIGMA, PI])
        ''' run Bayes Rule to update hypothesis '''
        currentBIC = bayes_info_criterion(image_matrix, PI, MU, SIGMA, k)
        ''' get log likelihood '''
        currentLoglike = loglikelihood(image_matrix, PI, MU, SIGMA, k)


        if currentBIC < BIC:
            BIC = currentBIC
            bic_index = k

        if currentLoglike > LOG_LIKE:
            LOG_LIKE = currentLoglike
            loglike_index = k

    return bic_index, loglike_index
