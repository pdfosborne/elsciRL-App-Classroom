import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Engine:
    def __init__(self, local_setup_info:dict={}) -> None:
        """Defines the environment function from the generator engine.
        Expects the following:
            - reset() to reset the env a start position(s)
            - step() to make an action and update the game state
            - legal_moves_generator() to generate the list of legal moves
        """
        # Ledger of the environment with meta information for the problem
        ledger_required = {
            'id': 'Unique Problem ID',
            'type': 'Language/Numeric',
            'description': 'Problem Description',
            'goal': 'Goal Description'
            }
        
        ledger_optional = {
            'reward': 'Reward Description',
            'punishment': 'Punishment Description (if any)',
            'state': 'State Description',
            'constraints': 'Constraints Description',
            'action': 'Action Description',
            'author': 'Author',
            'year': 'Year',
            'render_data':{'render_mode':'rgb_array', 
                           'render_fps':4}
        }
        ledger_gym_compatibility = {
            # Limited to discrete actions for now, set to arbitrary large number if uncertain
            'action_space_size':4, 
        }
        self.ledger = ledger_required | ledger_optional | ledger_gym_compatibility
        # --------------------------
        self.classroom_id = local_setup_info['classroom_id']
        if self.classroom_id == 'A':
            # Size of room
            self.x_range = [0,5]
            self.y_range = [0,5]
            # Define Class A
            #------------------------------------------------------------------------------
            # Adding example classroom to initialise environment
            ## 0. Create a copy of the initialized environment
            ## 1. Add state id name to x,y position
            ## 2. Add probability of command being followed to x,y position
            #----------------------------
            ## Define the x,y position of each state (manually and fixed for now)
            self.start_state_list = [[4,1],[3,1],[2,1],[1,1],[1,2],[1,3],[2,3],[3,2]]
            #self.start_state_list = [[1,1]]
            self.x_list = [4,3,2,1,1,1,2,3,3,4,4]
            self.y_list = [1,1,1,1,2,3,3,3,2,3,2]
            self.terminal_states = ['4_2', '4_3']
            self.rewards = [0,0,0,0,0,0,0,0,0,1,-1]
            ## Define the probability of each student following commands (manually and fixed for now)
            ### NOTE: 'Trap' states are defined by a 0 probability and are those for which the paper cannot move from (e.g. bins)
            self.state_probs = [0.4, 0.6, 0.5, 0.8, 0.6, 0.9, 0.9, 1, 0.2, 0, 0]
            ## Define state ids for our reference, not to be used by agent directly
            #self.state_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'M', 'recycling', 'trash']
            #----------------------------

        # Define 'universe' of positions and actions
        # ['left', 'right', 'up', 'down']
        self.legal_actions = [0,1,2,3]
        self.Classrooms = {}
        # Initialise complete classroom setup
        self.classroom_init = pd.DataFrame()
        for n1,x in enumerate(range(self.x_range[0], self.x_range[1])):
            for n2,y in enumerate(range(self.y_range[0], self.y_range[1])):
                state = str(x) + "_" + str(y)
                state_x = x
                state_y = y
                prob = 'na'
                reward = 'na'
                terminal = False
                df_row = pd.DataFrame({'state':state, 'state_x':state_x, 'state_y':state_y, 'prob':prob, 'reward':reward, 'terminal':terminal}, 
                                                                              index = [(n1*(self.y_range[1]-self.y_range[0])) + n2])
                self.classroom_init = pd.concat([self.classroom_init,df_row])

        current_classroom = self.classroom_init.copy()
        for item in range(0,len(self.state_probs)):
            #state_id = self.state_ids[item]
            state_prob = self.state_probs[item]
            state_reward = self.rewards[item]
            x = self.x_list[item]
            y = self.y_list[item]
            if str(x)+'_'+str(y) in self.terminal_states:
                terminal = True
            else:
                terminal = False

            # current_classroom['state_ids'] = np.where((current_classroom['state_x']==x)&(current_classroom['state_y']==y),
            #                                             state_id,
            #                                             current_classroom['state_ids'])
            current_classroom['prob'] = np.where((current_classroom['state_x']==x)&(current_classroom['state_y']==y),
                                                        state_prob,
                                                        current_classroom['prob'])
            current_classroom['reward'] = np.where((current_classroom['state_x']==x)&(current_classroom['state_y']==y),
                                                        state_reward,
                                                        current_classroom['reward'])
            current_classroom['terminal'] = np.where((current_classroom['state_x']==x)&(current_classroom['state_y']==y),
                                                        terminal,
                                                        current_classroom['terminal'])
        # Add this classroom to Classroom dictionary
        self.Classrooms['Classroom_'+str(self.classroom_id)] = current_classroom 

        # Initialize history
        self.action_history = []
        self.obs_history = []

    @staticmethod
    def action_outcome(state_x,state_y,action,states_df):
        # Produces x and y directional vectors for the action given the current x,y position
        # If this produces an outcome where there is no state (i.e. an empty slot) then the position won't change
        # Define basic action outcomes
        # ['left', 'right', 'up', 'down']
        if action == 0:
            u = -1
            v = 0
        elif action == 1:
            u = 1
            v = 0
        elif action == 2:
            u = 0
            v = 1
        elif action == 3:
            u = 0
            v = -1
        else:
            print("Error: Invalid action given")

        # Define overrides now based on max class ranges
        new_x = state_x + u
        new_y = state_y + v
        states_df_state = states_df[(states_df['state_x']==state_x)&(states_df['state_y']==state_y)]
        states_df_new_state = states_df[(states_df['state_x']==new_x)&(states_df['state_y']==new_y)]
        # If current state has probability 0, then this is trap state and we do not move
        if states_df_state['prob'].iloc[0] == 0:
            u = 0
            v = 0
        # If next state doesn't exist, don't move
        elif len(states_df_new_state) == 0:
            u = 0
            v = 0
        elif states_df_new_state['prob'].iloc[0] == 'na':
            u = 0
            v = 0
        # Otherwise, a wall is hit and paper doesn't move from current state
        elif (new_x==state_x) & (new_y==state_y):
            u = 0
            v = 0
        # If this returns a valid result, outcome acceptable
        else:
            u = u
            v = v
        return (u, v)


    def reset(self, start_obs:any=None):
        # Start episode position
        # env_reset_obs = random.choice(self.start_state_list)
        # start_obs = str(env_reset_obs[0]) + "_" + str(env_reset_obs[1])
        start_obs = "4_1" # For simplicity, always start at 4_1
        self.obs_history.append(start_obs)
        return start_obs 

    
    def step(self, state:any=None, action:any=None):
        #classroom_id, state_x, state_y, action):
        state_x = int(state.split('_')[0])
        state_y = int(state.split('_')[1])
        classroom = self.Classrooms['Classroom_'+str(self.classroom_id)]
        # Find current state and given probability
        state_data = classroom[(classroom['state_x']==state_x)&(classroom['state_y']==state_y)]
        prob = state_data['prob'].iloc[0]

        # Take action as successful or pick another random action if not
        action_rng = np.random.rand()
        if action_rng <= prob:
            action = action
        else:
            action_sub_list = self.legal_actions.copy()
            action_sub_list.remove(action)
            action = random.choice(action_sub_list)

        # TODO: move this into elsciRL agents
        if isinstance(action, np.int64):
            self.action_history.append(action.item())
        elif isinstance(action, np.ndarray):
            self.action_history.append(action.item())
        else:
            self.action_history.append(action)

        # Find movement direction given current state and action that ended up being taken
        current_action_outcome = Engine.action_outcome(state_x, state_y, action, classroom)
        u = current_action_outcome[0]
        v = current_action_outcome[1]
        next_state_x = state_x + u
        next_state_y = state_y + v
       
        next_state_data = classroom[(classroom['state_x']==next_state_x)&(classroom['state_y']==next_state_y)]
        reward = next_state_data['reward'].iloc[0]
        terminated = next_state_data['terminal'].iloc[0]
        next_state = str(next_state_x) + "_" + str(next_state_y)
        self.obs_history.append(str(next_state_x) + "_" + str(next_state_y))

        info = None

        return next_state, reward, terminated, info
    
    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        return self.legal_actions
    
    def render(self, state:any=None):
        """Render the environment as a gridworld."""
        if state is None:
            state = self.obs_history[-1]
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get current classroom data
        classroom = self.Classrooms['Classroom_'+str(self.classroom_id)]
        
        # Draw grid
        for x in range(self.x_range[0], self.x_range[1]):
            for y in range(self.y_range[0], self.y_range[1]):
                ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False))
                
        # Color terminal states
        for term_state in self.terminal_states:
            x, y = map(int, term_state.split('_'))
            if classroom[(classroom['state_x']==x) & (classroom['state_y']==y)]['reward'].iloc[0] > 0:
                color = 'green'  # positive reward
            else:
                color = 'red'  # negative reward
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color, alpha=0.5))
        
        # Highlight current state
        curr_x, curr_y = map(int, state.split('_'))
        ax.add_patch(plt.Rectangle((curr_x, curr_y), 1, 1, color='blue', alpha=0.5))
        
        # Add probability text to cells
        for _, row in classroom.iterrows():
            current_state = '['+str(row["state_x"])+','+str(row['state_y'])+']'
            if (row['reward'] != 'na'):
                if row['reward'] == 0:
                    if state == str(row['state_x'])+'_'+str(row['state_y']):
                        ax.text(row['state_x'] + 0.5, row['state_y'] + 0.5, 
                            f'{current_state}', ha='center', va='center',
                            fontsize=30)
                    else:
                        ax.text(row['state_x'] + 0.5, row['state_y'] + 0.5, 
                            f'{current_state}', ha='center', va='center',
                            fontsize=16)
                else:
                    ax.text(row['state_x'] + 0.5, row['state_y'] + 0.5, 
                        f'r={row["reward"]}', ha='center', va='center',
                        fontsize=30)
        
        # Set grid properties
        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.y_range[0], self.y_range[1])
        ax.grid(True)
        plt.show()

        return fig
    
    def close(self):
        """Close/Exit the environment."""
        self.Environment.close()
