import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


#Code used to set up the display, as according to the tutorial
#________________________________________________________________________________________________________________
env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#________________________________________________________________________________________________________________






#We construct the replay memory of each new state, which will be sampled from the DQN algorithm in order to calculate Huber Loss.
#We will minimize Huber Loss in order to update our training such that our training follows the Bellman equation.
#The Bellman equation states the basic update rule for Q-learning
#________________________________________________________________________________________________________________
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
#________________________________________________________________________________________________________________




#We will utilize DQN over Q-learning. This is because we are using images as our state space. 
#Each difference of images is approximately of size 90x40 pixels, or 3600 pixels total.
#Each pixel is an RGB that consists of 3 colors, each with 256 possible values, or 256^3 possibilities
#Thus, our entire state space has a total value of 3600^256^3 possible states.
#Because this is unreasonable for regular tabular Q-learning to store that many states, 
#we will use DQN to create a function estimator in order to generalize across states.

#Our action space consists of a +1 or a -1 force in the horizontal plane to control the pole-cart.

#Our environment award will be +1 for every timestep that passes with the pole being upright,
#while the episode will end once the pole is more than 15 units from the vertical, or the cart is more than 2.4 units from the center









#This code block represents the CNN that we will be using and back propogating through in order to create our 
#function approximator for the best action for a given state input.

#For each key line of code, please give a brief description of the RL algorithm that is implemented
#________________________________________________________________________________________________________________
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        #Our neural network will consist of 3 layers of Convolution, with the end result having 32 output channels instead of the 3 input channels of color
        #Because our kernel_size is not 1 for each layer, as we convolve, our convolution image will become smaller than the original 90x40
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        #This line of code calculates the size of the output after all 3 convolutions have passed
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
#________________________________________________________________________________________________________________










        
#This code block is used to obtain the screen of which we are observing our cart-pole, and thus the source of our input images for states
#________________________________________________________________________________________________________________
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#             interpolation='none')
# plt.title('Example extracted screen')
# plt.show()
#________________________________________________________________________________________________________________
















#This is the code block where we set up the model, the optimizer, and some other utilities

#For each key line of code, please give a brief description of the RL algorithm that is implemented


#________________________________________________________________________________________________________________
#Batch size will control the number of sampled previous transitions that we will use to train the network
BATCH_SIZE = 128
#Gamma is the discount factor. Because it is close to 1, we will capture long-term effectiveness well over the short-term
GAMMA = 0.999
#Epsilon will start high so that we can explore random actions. This will decrease over time as in the long term, 
#exploration becomes less important as we are trying to optimize reward.
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
#Target update controls how often states are updated in terms of weight/reward to be used by the training 
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n



#We create two DQNs.
#Policy_net will be the DQN that holds the action space of the current state, including the q-value for each action
#Target_net is a DQN that we will use in order to obtain Q values of each next state of a certain current state
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


#initializes the optimizer and replay memory
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

#This function is called whenever we request an action for each timestep
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    #selects an action based on random chance and the epsilon threshold
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.y
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        #selects a random action
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []
mean_durations = []
for i in range(3):
    mean_durations.append(0)

def plot_durations(numFig):
    plt.figure(numFig)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    if numFig == 2:
        plt.title('No Training')
    elif numFig == 3:
        plt.title('100 Episode Training')
    elif numFig == 4:
        plt.title('200 Episode Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 30:
        means = durations_t.unfold(0, 30, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(29), means))
        mean_durations[numFig-2] = sum(episode_durations[-30:]) / 30
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
#________________________________________________________________________________________________________________




#This is the code block that sets up the training for our model
#For each key line of code, please give a brief description of the RL algorithm that is implemented
#________________________________________________________________________________________________________________
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
#________________________________________________________________________________________________________________
    





#This is the code block that will be executed in order to train the model. 
#________________________________________________________________________________________________________________

trainingTimes = []
for i in range(3):
    trainingTimes.append(0)


#This is the hyperparameter in order to determine how may episodes of training we should go through
#For short training, we will do 100 episodes
num_episodes = 100
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    #Initialize the screens necessary for obtaining states
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        #We are done if we reach a poin that ends the simulation, when the pole is 15 degrees off or the cart is 2.4 from the center
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            trainingTimes[1] += t + 1
            plot_durations(3)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())






#reset
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0
episode_durations = []

#For long training, we will do 200 episodes
num_episodes = 200
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    #Initialize the screens necessary for obtaining states
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        #We are done if we reach a poin that ends the simulation, when the pole is 15 degrees off or the cart is 2.4 from the center
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            trainingTimes[2] += t + 1
            plot_durations(4)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())





#reset
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0
episode_durations = []

#In order to determine how the agent will do without training, we will remove the optimize_model() function, and take the mean of 30 episodes of no training
num_episodes = 100
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    #Initialize the screens necessary for obtaining states
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        #We are done if we reach a poin that ends the simulation, when the pole is 15 degrees off or the cart is 2.4 from the center
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        if done:
            episode_durations.append(t + 1)
            trainingTimes[0] += t + 1
            plot_durations(2)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


plt.figure(5)
plt.clf()
plt.title('Performance Given Training')
plt.xlabel('Type of Training')
plt.ylabel('Average Duration of last 30 episodes')
trainingType = ['No Training', '100 Episodes', '200 Episodes']
plt.plot(trainingType, mean_durations, linestyle = None)

plt.figure(6)
plt.clf()
plt.title('Total Duration of each Training Type')
plt.xlabel('Type of Training')
plt.ylabel('Total Duration')
trainingType = ['No Training', '100 Episodes', '200 Episodes']
plt.plot(trainingType, trainingTimes, linestyle = None)



print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
#________________________________________________________________________________________________________________


#As can be seen from figure 5, a training time of 200 episodes is clearly superior in terms of average reward/duration
#Even training times as little as 100 episodes is still better than an agent that just moves randomly
#HOwever, higher rewards do come at the cost of processing time, as seen from figure 6.