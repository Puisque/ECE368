import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    # forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps

    # first compute the forward and backward messages
    # loop through all the time_steps
    for i in range(num_time_steps):
        # create Distribution dictionaries
        end = num_time_steps - 1 - i
        forward_messages[i] = rover.Distribution({})
        backward_messages[end] = rover.Distribution({})
        # Initialization
        if i == 0:
            # TODO: initialize forward message to be p(z0) * p( (x0,y0)| z0 )
            for zn in prior_distribution:
                prior_z0 = prior_distribution[zn]
                z0 = observations[i]
                pcond_X0_z0 = observation_model(zn)[z0]
                if pcond_X0_z0 != 0:
                    forward_messages[i][zn] = prior_z0 * pcond_X0_z0
            # renormalize forward message
            forward_messages[i].renormalize()

            # TODO: initialize the backward message to be 1 at state all states
            for zn in all_possible_hidden_states:
                backward_messages[end][zn] = 1

        # Forward/Backward Messages
        else:
            # TODO: Compute the forward messages
            xn = observations[i]
            for zn in all_possible_hidden_states:

                # Part 2: check if observation is None
                if xn == None:
                    # set conditional probability to be 1
                    pcond_Xn_zn = 1
                else:
                    pcond_Xn_zn = observation_model(zn)[xn]

                if pcond_Xn_zn != 0:
                    sum = 0
                    for zn_prev in (forward_messages[i-1]):
                        forward_prev = forward_messages[i-1][zn_prev]
                        pcond_zn_zn_prev = transition_model(zn_prev)[zn]
                        sum += forward_prev * pcond_zn_zn_prev
                    if sum != 0:
                        forward_messages[i][zn] = pcond_Xn_zn * sum
            # renormalize forward message
            forward_messages[i].renormalize()

            # TODO: Compute the backward messages
            # end denotes the current state (n) while end+1 denotes the state after it (n_aft)
            xn_aft = observations[end + 1]
            for zn in all_possible_hidden_states:
                sum = 0
                for zn_aft in backward_messages[end+1]:

                    # Part 2: check if observation is None
                    if xn_aft == None:
                        # set conditional probability to be 1 if so
                        pcond_Xn_aft_zn_aft = 1
                    else:
                        pcond_Xn_aft_zn_aft = observation_model(zn_aft)[xn_aft]

                    beta_zn_aft = backward_messages[end + 1][zn_aft]
                    pcond_zn_aft_zn = transition_model(zn)[zn_aft]
                    sum += beta_zn_aft * pcond_zn_aft_zn * pcond_Xn_aft_zn_aft
                if sum != 0:
                    backward_messages[end][zn] = sum

    # TODO: Compute the marginals
    # loop through all the time_steps
    for i in range(num_time_steps):
        # create Distribution dictionaries
        marginals[i] = rover.Distribution()
        # calculate the numerator and denominator (normalization factor)
        sum_alpha_beta = 0
        for zn in all_possible_hidden_states:
            alpha = forward_messages[i][zn]
            beta = backward_messages[i][zn]
            if alpha * beta != 0:
                marginals[i][zn] = (alpha * beta)
            sum_alpha_beta += forward_messages[i][zn] * backward_messages[i][zn]
        # check if non-zero probability (for normalization)
        if sum_alpha_beta != 0:
            # normalize each zn
            for zn in marginals[i].keys():
                marginals[i][zn] = marginals[i][zn] / sum_alpha_beta
    # print(marginals)
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here

    num_time_steps = len(observations)
    forward_pass = [None] * num_time_steps
    w = [None] * num_time_steps
    max_w_zn = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps

    # loop through all the time_steps
    for i in range(num_time_steps):
        # create Distribution dictionaries
        w[i] = rover.Distribution()
        if i == 0:
            # TODO: initialize forward message to be ln(p(z0) * p( (x0,y0)| z0 ))
            for zn in prior_distribution:
                prior_z0 = prior_distribution[zn]
                z0 = observations[i]
                pcond_X0_z0 = observation_model(zn)[z0]
                if prior_z0*pcond_X0_z0 != 0:
                    w[i][zn] = np.log(prior_z0 * pcond_X0_z0)
        else:
            # TODO: Compute the forward messages
            xn = observations[i]
            max_w_zn[i] = dict()
            for zn in all_possible_hidden_states:
                # Part 2: check if observation is None
                if xn == None:
                    # set conditional probability to be 1
                    pcond_Xn_zn = 1
                else:
                    pcond_Xn_zn = observation_model(zn)[xn]
                if pcond_Xn_zn != 0:
                    # set maximum value to be -inf, and find the maximum of all values
                    max_zn_prev = - np.inf
                    # iterate through all the previous zn, find the max value, and save it
                    for zn_prev in (w[i - 1]):
                        pcond_zn_zn_prev = transition_model(zn_prev)[zn]
                        # avoid log of 0 case
                        if pcond_zn_zn_prev != 0:
                            w_prev = w[i - 1][zn_prev]
                            new_zn_prev = np.log(pcond_zn_zn_prev) + w_prev
                            if new_zn_prev > max_zn_prev:
                                max_zn_prev = new_zn_prev
                                # store the maximum path to get to a certain zn from zn_prev
                                max_w_zn[i][zn] = zn_prev
                    # set the zn to be ln (p( (xn,yn)| zn )) + max {ln(p(zn,zn_prev)) +  w(zn_prev)}
                    w[i][zn] = np.log(pcond_Xn_zn) + max_zn_prev

    # loop through all the time_steps to determine optimal path
    for i in range(num_time_steps):
        end = num_time_steps - 1 - i
        if i == 0:
            # find the most probable zn at the end
            max_zn = - np.inf
            zn_star = None
            for zn in w[end]:
                zn_temp = zn
                max_zn_temp = w[end][zn_temp]
                if max_zn_temp > max_zn:
                    max_zn = max_zn_temp
                    zn_star = zn_temp
            estimated_hidden_states[end] = zn_star
        else:
            # find the path the links the old zn_star to the current time
            zn_star_prev = estimated_hidden_states[end + 1]
            estimated_hidden_states[end] = max_w_zn[end + 1][zn_star_prev]

    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    # missing_observations = False
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    # Part 4 Errors

    # compute the estimated forward backward true hidden states, by taking the argmax of each state zn
    estimate_forward_back = [None] * num_time_steps
    for i in range(num_time_steps):
        max_prob = 0
        for zn in marginals[i]:
            temp_prob = marginals[i][zn]
            if temp_prob > max_prob:
                max_prob = temp_prob
                estimate_forward_back[i] = zn

    error_count_forward_back = 0
    error_count_viterbi = 0
    for i in range(num_time_steps):
        if estimate_forward_back[i] != hidden_states[i]:
            error_count_forward_back += 1
        if estimated_states[i] != hidden_states[i]:
            error_count_viterbi += 1
    print("Viterbi Error Probability (z_tilde): {} | Forward-Backward Error Probability (z_check): {}"
          .format(error_count_viterbi/num_time_steps, error_count_forward_back/num_time_steps))

    # Part 5 Valid Sequence
    for i in range(num_time_steps-1):
        print('z_{}: state:{}, action: {}'.format(i, estimate_forward_back[i][0:2], estimate_forward_back[i+1][2]))
    print('z_{}: state:{}'.format(i, estimate_forward_back[i][0:2]))
    '''
    z_0: state:(10, 0), action: down
    z_1: state:(10, 1), action: down
    z_2: state:(10, 2), action: down
    z_3: state:(10, 3), action: down
    z_4: state:(10, 4), action: down
    z_5: state:(10, 5), action: down
    z_6: state:(10, 6), action: down
    z_7: state:(10, 7), action: stay
    z_8: state:(10, 7), action: stay
    z_9: state:(10, 7), action: stay
    z_10: state:(10, 7), action: right
    z_11: state:(11, 7), action: stay
    z_12: state:(11, 7), action: left
    z_13: state:(10, 7), action: left
    z_14: state:(9, 7), action: left
    z_15: state:(8, 7), action: left
    z_16: state:(7, 7), action: left
    z_17: state:(6, 7), action: left
    z_18: state:(5, 7), action: left
    z_19: state:(4, 7), action: left
    z_20: state:(3, 7), action: left
    z_21: state:(2, 7), action: left
    z_22: state:(1, 7), action: left
    z_23: state:(0, 7), action: stay
    z_24: state:(0, 7), action: right
    z_25: state:(1, 7), action: right
    z_26: state:(2, 7), action: right
    z_27: state:(3, 7), action: right
    z_28: state:(4, 7), action: right
    z_29: state:(5, 7), action: right
    z_30: state:(6, 7), action: right
    z_31: state:(7, 7), action: right
    z_32: state:(8, 7), action: right
    z_33: state:(9, 7), action: right
    z_34: state:(10, 7), action: right
    z_35: state:(11, 7), action: stay
    z_36: state:(11, 7), action: stay
    z_37: state:(11, 7), action: stay
    z_38: state:(11, 7), action: stay
    z_39: state:(11, 7), action: left
    z_40: state:(10, 7), action: left
    z_41: state:(9, 7), action: left
    z_42: state:(8, 7), action: left
    z_43: state:(7, 7), action: left
    z_44: state:(6, 7), action: left
    z_45: state:(5, 7), action: stay
    z_46: state:(5, 7), action: right
    z_47: state:(6, 7), action: right
    z_48: state:(7, 7), action: right
    z_49: state:(8, 7), action: right
    z_50: state:(9, 7), action: right
    z_51: state:(10, 7), action: right
    z_52: state:(11, 7), action: stay
    z_53: state:(11, 7), action: left
    z_54: state:(10, 7), action: left
    z_55: state:(9, 7), action: left
    z_56: state:(8, 7), action: left
    z_57: state:(7, 7), action: left
    z_58: state:(6, 7), action: left
    z_59: state:(5, 7), action: left
    z_60: state:(4, 7), action: stay
    z_61: state:(4, 7), action: stay
    z_62: state:(4, 7), action: left
    z_63: state:(3, 7), action: stay
    z_64: state:(3, 7), action: stay        <- here, the state is (3,7) and the action should be stay
    z_65: state:(2, 7), action: right       <- but here, the state changes to (2,7)
    z_66: state:(3, 7), action: stay
    z_67: state:(3, 7), action: up
    z_68: state:(3, 6), action: up
    z_69: state:(3, 5), action: up
    z_70: state:(3, 4), action: up
    z_71: state:(3, 3), action: stay
    z_72: state:(3, 3), action: right
    z_73: state:(4, 3), action: right
    z_74: state:(5, 3), action: right
    z_75: state:(6, 3), action: right
    z_76: state:(7, 3), action: right
    z_77: state:(8, 3), action: right
    z_78: state:(9, 3), action: right
    z_79: state:(10, 3), action: right
    z_80: state:(11, 3), action: stay
    z_81: state:(11, 3), action: up
    z_82: state:(11, 2), action: up
    z_83: state:(11, 1), action: up
    z_84: state:(11, 0), action: stay
    z_85: state:(11, 0), action: down
    z_86: state:(11, 1), action: down
    z_87: state:(11, 2), action: down
    z_88: state:(11, 3), action: down
    z_89: state:(11, 4), action: down
    z_90: state:(11, 5), action: down
    z_91: state:(11, 6), action: down
    z_92: state:(11, 7), action: stay
    z_93: state:(11, 7), action: stay
    z_94: state:(11, 7), action: left
    z_95: state:(10, 7), action: left
    z_96: state:(9, 7), action: left
    z_97: state:(8, 7), action: left
    z_98: state:(7, 7), action: left
    z_98: state:(7, 7)
    '''

    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
