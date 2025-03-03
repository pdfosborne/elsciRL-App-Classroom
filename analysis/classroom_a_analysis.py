import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@staticmethod
def vel(theta, theta_0=0, theta_dead=np.pi / 12):
    return 1 - np.exp(-(theta - theta_0) ** 2 / theta_dead)

@staticmethod
def rew(theta, theta_0=0, theta_dead=np.pi / 12):
    return vel(theta, theta_0, theta_dead) * np.cos(theta)


class Analysis:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        
    def trace_plot(self):
        """Define experiment directories to run and override names for
        improved plot titles."""
        # TODO MOVE ALL THIS FOLDER SEARCHING INTO ELSCIRL
        path = self.save_dir 
        path_folders = os.listdir(path)
        plot_list = {}
        for n,folder in enumerate(path_folders):
            if os.path.isdir(path+'/'+folder):
                exp_path = path + '/' + folder
                exp_path_folders = os.listdir(exp_path)

                count_check = 0
                previous_agent = ''
                policy_output_dict = {}
                policy_list = []
                for result_folders in exp_path_folders:
                    if os.path.isdir(exp_path+'/'+result_folders):
                        agent = result_folders.split('__')[0]
                        if 'training' in result_folders:
                            # Update output dict with policy list
                            if (agent != previous_agent) and (previous_agent != ''):
                                print("Agent: ", previous_agent)
                                print("Policy List: ", policy_list)
                                if previous_agent not in policy_output_dict.keys():
                                    policy_output_dict[previous_agent] = policy_list
                                else:
                                    prior_policy_list = policy_output_dict[previous_agent]
                                    prior_policy_list.append(policy_list[0])
                                    policy_output_dict[previous_agent] = prior_policy_list
                                policy_list = []
                            testing_results_path = exp_path + '/' + result_folders
                            results = pd.read_csv(testing_results_path+"/results.csv")
                            
                            policy = results['action_history'].mode()[0]
                            policy_fix = policy.split(',')
                            policy_fix = [int(i.replace('[4','3').replace('[2','2').replace('[1','1').replace('[0','0').replace('3]','3').replace('2]','2').replace('1]','1').replace('0]','0')) for i in policy_fix]
                            policy_list.append(policy_fix)
                            count_check += 1
                            previous_agent = agent
                # Add final policy list to output dict
                if previous_agent not in policy_output_dict.keys():
                    policy_output_dict[previous_agent] = policy_list
                else:
                    prior_policy_list = policy_output_dict[previous_agent]
                    prior_policy_list.append(policy_list[0])
                    policy_output_dict[previous_agent] = prior_policy_list
                print(policy_output_dict)
                # Plotting
                for agent in policy_output_dict.keys():
                    figure = plt.figure()
                    ax = figure.add_subplot(1, 1, 1)
                    #print(count_check)
                    # Re-applies actions made by agent to observe path
                    if 'instr' in folder.lower():
                        exp_title = path.split('/')[-1] + ' - ' + agent
                    else:
                        exp_title = folder + ' - ' + agent