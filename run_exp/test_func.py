import numpy as np

# ----- Constants -----
num_bandits = 3
block_len = 40
num_blocks = 6
num_trials = block_len * num_blocks * 2
init_payoff = [60, 30, 10]
decayTheta = init_payoff.copy()
payoff_bounds = [5, 95]
decay_lambda = 0.6
drift_noise = 8

rotation_trials = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440]
deterministic_trials = [30, 70, 110, 150, 190]

# Add bumped deterministic trials
ctx_bump = 2
for i in range(len(deterministic_trials)):
    for j in range(1, ctx_bump):
        trial_to_add = deterministic_trials[i] + j
        deterministic_trials.append(trial_to_add)

# ----- Functions -----
def normal_random():
    return np.random.normal(loc=0, scale=1)

def mean(x):
    return sum(x) / len(x)

def rotate_decayTheta(current_decayTheta):
    bestOpt = np.argmax(current_decayTheta)
    shuffled = current_decayTheta.copy()
    while np.argmax(shuffled) == bestOpt:
        np.random.shuffle(shuffled)
    return list(shuffled)

# ----- Initialize payout matrix -----
payout = np.zeros((num_bandits, num_trials))
for i in range(num_bandits):
    payout[i][0] = init_payoff[i]

# ----- Main drift loop -----
for trial_idx in range(num_trials):

    # ROTATION TRIALS: Change which bandit has the highest expected payoff
    if trial_idx in rotation_trials:
        decayTheta = rotate_decayTheta(decayTheta)

    # DETERMINISTIC TRIALS: Force the best bandit's value to 100
    if trial_idx in deterministic_trials:
        bestOpt = np.argmax(decayTheta)
        payout[bestOpt][trial_idx] = 100

    # DRIFT: Smooth payoff change from previous value
    if trial_idx > 0:
        decay_lambda_eff = 0.95 if trial_idx % block_len < 2 else decay_lambda

        for bandit in range(num_bandits):
            prev = payout[bandit][trial_idx - 1]
            theta = decayTheta[bandit]
            noise = normal_random() * drift_noise

            drifted = decay_lambda_eff * prev + (1 - decay_lambda_eff) * theta + noise

            # Reflect at bounds
            if drifted > payoff_bounds[1]:
                drifted = payoff_bounds[1] - (drifted - payoff_bounds[1])
            elif drifted < payoff_bounds[0]:
                drifted = payoff_bounds[0] + (payoff_bounds[0] - drifted)

            payout[bandit][trial_idx] = drifted

# Convert to list of lists if needed
payout_list = payout.tolist()

print(len(payout_list[0]))
reward = 2
no_reward = 20

practice_rewards = {
    1: [reward, reward, reward, no_reward, reward, reward, reward, reward, no_reward, reward],
    2: [no_reward, reward, no_reward, reward, no_reward, no_reward, no_reward, reward, no_reward, no_reward],
    3: [no_reward, no_reward, reward, no_reward, no_reward, no_reward, reward, no_reward, reward, reward]
}

print(practice_rewards[3][1])
image_prefix = '../run_exp/static/images/'


# Instruction imgs
all_contexts = image_prefix + "contexts/all_contexts.png"
tutorial_ship = image_prefix + "tutorial/ship_center.png"
tutorial_pirates = image_prefix + "tutorial/pirates_all_crop.png"
tutorial_reward = image_prefix + "tutorial/reward.png"
tutorial_noreward = image_prefix + "tutorial/reward_no.png"
tutorial_blue_pirate = image_prefix + "tutorial/blue.png"
tutorial_all_pirates = image_prefix + "tutorial/all_pirates.png"

blue_win = image_prefix + "tutorial/blue_win.png"
black_win = image_prefix + "tutorial/black_win.png"
red_win = image_prefix + "tutorial/red_win.png"
white_win = image_prefix + "tutorial/white_win.png"
blue_no_win = image_prefix + "tutorial/blue_no_win.png"
black_no_win = image_prefix + "tutorial/black_no_win.png"
red_no_win = image_prefix + "tutorial/red_no_win.png"
white_no_win = image_prefix + "tutorial/white_no_win.png"
timeout_img = image_prefix + "miscellaneous/hurry_up.png"

example_probe = image_prefix + "tutorial/example_probe.png"
contingency = image_prefix + "tutorial/contingency.png"

# Practice Images
deck = image_prefix + "miscellaneous/deck.png"
ahoy = image_prefix + "travel/ahoy.png"
desert_welcome_text = image_prefix + "travel/welcome_desert.png"
cavern_welcome_text = image_prefix + "travel/welcome_cavern.png"
desert_img = image_prefix + "contexts/context_desert.png"
cavern_img = image_prefix + "contexts/context_cavern.png"

all_pirates = image_prefix + "pirates/pirates_all.png"
red_pirate = image_prefix + "pirates/red_beard.png"
white_pirate = image_prefix + "pirates/white_beard.png"
black_pirate = image_prefix + "pirates/black_beard.png"

reward = image_prefix + "rewards/reward.png"
no_reward = image_prefix + "rewards/no_reward.png"

probe_ship = image_prefix + "miscellaneous/cargo_ship.png"

# Practice Probes
practice_probes = [
  "probes/probes-256.png",
  "probes/probes-257.png",
  "probes/probes-258.png",
  "probes/probes-259.png",
  "probes/probes-260.png",
  "probes/probes-261.png",
  "probes/probes-262.png",
  "probes/probes-263.png",
  "probes/probes-264.png",
  "probes/probes-265.png"
]
# Stacked images
desert_welcome = [deck,desert_img,ahoy,desert_welcome_text]
desert_pirates = [deck,desert_img,all_pirates]
desert_red = [deck,desert_img,red_pirate]
desert_white = [deck,desert_img,white_pirate]
desert_black = [deck,desert_img,black_pirate]

desert_red_remember = []
desert_red_reward = []
desert_white_remember = []
desert_white_reward = []
desert_black_remember = []
desert_black_reward = []

cavern_welcome = [deck,cavern_img,ahoy,cavern_welcome_text]
cavern_pirates = [deck,cavern_img,all_pirates]
cavern_red = [deck,cavern_img,red_pirate]
cavern_white = [deck,cavern_img,white_pirate]
cavern_black = [deck,cavern_img,black_pirate]

cavern_red_remember = []
cavern_red_reward = []
cavern_white_remember = []
cavern_white_reward = []
cavern_black_remember = []
cavern_black_reward = []

practice_rewards = {
    1: [reward, reward, reward, no_reward, reward, reward, reward, reward, no_reward, reward],
    2: [no_reward, reward, no_reward, reward, no_reward, no_reward, no_reward, reward, no_reward, no_reward],
    3: [no_reward, no_reward, reward, no_reward, no_reward, no_reward, reward, no_reward, reward, reward]
}

for i,probe in enumerate(practice_probes):
    desert_red_remember.append([deck,desert_img,red_pirate,probe_ship,image_prefix+probe])
    desert_red_reward.append([deck,desert_img,red_pirate,probe_ship,image_prefix+probe,practice_rewards[1][i]])
    desert_white_remember.append([deck,desert_img,white_pirate,probe_ship,image_prefix+probe])
    desert_white_reward.append([deck,desert_img,white_pirate,probe_ship,image_prefix+probe,practice_rewards[2][i]])
    desert_black_remember.append([deck,desert_img,black_pirate,probe_ship,image_prefix+probe])
    desert_black_reward.append([deck,desert_img,black_pirate,probe_ship,image_prefix+probe,practice_rewards[3][i]])


print(desert_black_remember[0])

import matplotlib.pyplot as plt

# Assume `payout` is a 2D list or NumPy array: shape (3, num_trials)
# If you haven't already defined `payout`, run the generation code first.

# Plotting
plt.figure(figsize=(12, 6))

# # Plot each bandit's payout trajectory
# plt.plot(payout[0][0:], label='Bandit 1', color='green')
# plt.plot(payout[1][0:], label='Bandit 2', color='red')
# plt.plot(payout[2][0:], label='Bandit 3', color='blue')
# for d in [30, 70, 110, 150, 190]:
#     print(d)
#     plt.axvspan(d,d+10, color='pink', alpha=0.3)
#     plt.axvline(d,color='black')
# # Formatting
# plt.xlabel('Trial')
# plt.ylabel('Reward Value')
# plt.title('Bandit Payouts Over Trials')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

probabilities = {
    1: payout[0] * 0.01,
    2: payout[1] * 0.01,
    3: payout[2] * 0.01
}

#Plotting
# plt.figure(figsize=(12, 6))

# # Plot each bandit's payout trajectory
# plt.plot(list(range(241, 481)),probabilities[1][240:], label='Bandit 1', color='green')
# plt.plot(list(range(241, 481)),probabilities[2][240:], label='Bandit 2', color='red')
# plt.plot(list(range(241, 481)),probabilities[3][240:], label='Bandit 3', color='blue')
# for d in [240, 280, 320, 360, 400,440]:
#     print(d)
#     plt.axvline(d,color='black')
# # Formatting
# plt.xlabel('Trial')
# plt.xticks([240,480])
# plt.ylabel('Probability for Reward')
# plt.title('Bandit Payout Probabilities Over Trials')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
import random
valid_probe_images = []
invalid_probe_images = []
for i in range(1,231): # 230 trials (30 trials first block + 40 trials * 5 other blocks)
    if i < 10:   
        valid_probe_images.append(image_prefix + f"probes-0{i}.png")
    else:
        valid_probe_images.append(image_prefix+ f"probes-{i}.png")
for i in range(231,256): # 25 trials (wrong probe trials during testing)
    invalid_probe_images.append(image_prefix +f"probes-{i}.png")

random.shuffle(valid_probe_images)
random.shuffle(invalid_probe_images)
print(len(probabilities[1]))




# Valid probes (seen in task) and Invalid (foils during memory phase)
valid_probe_images = []
invalid_probe_images = []
for i in range(1,231): # 230 trials (30 trials first block + 40 trials * 5 other blocks)
    if i < 10:   
        valid_probe_images.append(image_prefix + f"probes-0{i}.png")
    else:
        valid_probe_images.append(image_prefix + f"probes-{i}.png")
for i in range(231,256): # 25 trials (wrong probe trials during testing)
    invalid_probe_images.append(image_prefix + f"probes-{i}.png")

random.shuffle(valid_probe_images)
random.shuffle(invalid_probe_images)

# Context-based trial order
first_block = 30
contexts = [image_prefix + "contexts/context_coast.png",
            image_prefix + "contexts/context_countryside.png",
            image_prefix + "contexts/context_mountain.png",
            image_prefix + "contexts/context_forest.png",
            image_prefix + "contexts/context_highway.png",
            image_prefix + "contexts/context_city.png"]
random.shuffle(contexts)

stacked_all_pirates = []
stacked_red_remember = []
stacked_red_reward = []
stacked_white_remember = []
stacked_white_reward = []
stacked_black_remember = []
stacked_black_reward = []
context_labels = []

for context_idx,context in enumerate(contexts):
    if context_idx == 0:
        for trial_idx in range(first_block):
            stacked_all_pirates.append([deck,context,all_pirates])
            stacked_red_remember.append([deck,context,red_pirate,probe_ship,valid_probe_images[trial_idx]])
            stacked_red_reward.append([deck,context,red_pirate,probe_ship,valid_probe_images[trial_idx],probabilities[1][trial_idx]])
            stacked_white_remember.append([deck,context,white_pirate,probe_ship,valid_probe_images[trial_idx]])
            stacked_white_reward.append([deck,context,white_pirate,probe_ship,valid_probe_images[trial_idx],probabilities[2][trial_idx]])
            stacked_black_remember.append([deck,context,black_pirate,probe_ship,valid_probe_images[trial_idx]])
            stacked_black_reward.append([deck,context,black_pirate,probe_ship,valid_probe_images[trial_idx],probabilities[3][trial_idx]])
            context_labels.append(context.split("contexts/context_")[-1].split(".png")[0])
    else:
        for trial_idx in range(block_len):
            stacked_all_pirates.append([deck,context,all_pirates])
            stacked_red_remember.append([deck,context,red_pirate,probe_ship,valid_probe_images[trial_idx]])
            stacked_red_reward.append([deck,context,red_pirate,probe_ship,valid_probe_images[trial_idx],probabilities[1][trial_idx]])
            stacked_white_remember.append([deck,context,white_pirate,probe_ship,valid_probe_images[trial_idx]])
            stacked_white_reward.append([deck,context,white_pirate,probe_ship,valid_probe_images[trial_idx],probabilities[2][trial_idx]])
            stacked_black_remember.append([deck,context,black_pirate,probe_ship,valid_probe_images[trial_idx]])
            stacked_black_reward.append([deck,context,black_pirate,probe_ship,valid_probe_images[trial_idx],probabilities[3][trial_idx]])
            context_labels.append(context.split("contexts/context_")[-1].split(".png")[0])



contexts = [image_prefix + "contexts/context_coast.png",
            image_prefix + "contexts/context_countryside.png",
            image_prefix + "contexts/context_mountain.png",
            image_prefix + "contexts/context_forest.png",
            image_prefix + "contexts/context_highway.png",
            image_prefix + "contexts/context_city.png"]

welcomeArray = [image_prefix + "travel/welcome_coast.png",
                image_prefix + "travel/welcome_meadow.png",
                image_prefix + "travel/welcome_mountain.png",
                image_prefix + "travel/welcome_forest.png",
                image_prefix + "travel/welcome_road.png",
                image_prefix + "travel/welcome_city.png"]

indices = list(range(len(contexts)))
random.shuffle(indices)

contexts = [contexts[i] for i in indices]
welcomeArray = [welcomeArray[i] for i in indices]

ifReward = {key: np.random.binomial(1, prob) for key, prob in probabilities.items()}

reward_imgs = {
    key: np.where(arr == 1, 'reward', 'no_reward')
    for key, arr in ifReward.items()
}
# print(ifReward[3].tolist())


import numpy as np

# Parameters
sample_window = 10
mean_ct = 5
min_ct = 2
max_ct = 8
num_probes = sample_window * (num_blocks-1)
num_invalid_probes = np.round(num_probes/5)
num_probes = int(num_probes + num_invalid_probes)
# Step 1: Generate exponentially-distributed choice_blocks
log_rand = np.log(np.random.rand(num_probes))
log_rand_div_mean_ct = log_rand / (1 / mean_ct)
choice_blocks = np.ceil(log_rand_div_mean_ct) * -1 + min_ct
choice_blocks = np.clip(choice_blocks, min_ct, max_ct).astype(int)

# Step 2: Adjust until sum == num_trials/2 and all values are within bounds
while (choice_blocks.sum() != num_trials // 2 or
       np.any(choice_blocks < min_ct) or
       np.any(choice_blocks > max_ct)):

    # Random index to modify
    ind = np.random.randint(0, num_probes)

    # Increase or decrease based on current sum
    delta = int(np.sign(choice_blocks.sum() - (num_trials // 2)))
    choice_blocks[ind] -= delta

    # Clamp again
    choice_blocks = np.clip(choice_blocks, min_ct, max_ct)

# Step 3: Subtract 1 from each (to leave space for a mem_trial) and add 10 to first
choice_blocks = choice_blocks - 1
choice_blocks[0] += 10

# Step 4: Compute memory probe trial indices
block_sizes = choice_blocks + 1
cumsum_blocks = np.cumsum(block_sizes)
mem_probe_trials = cumsum_blocks + (num_trials // 2)  # shift into 2nd half
mem_probe_trials = mem_probe_trials - 1  # zero-indexing

# Step 5: Compute choice trials (set difference)
all_trials = np.arange(num_trials)
choice_trials = np.setdiff1d(all_trials, mem_probe_trials)

# # Print results
# print("choice_blocks:", choice_blocks.tolist())
# print("mem_probe_trials:", mem_probe_trials.tolist())
# print("choice_trials:", choice_trials.tolist())

response_check=np.random.randint(0,2,230)
response_sorted = []
memory_probes = []
final_memory_probes = []
probed_context = []
final_probed_context = []
def init_responses():
    global response_sorted,memory_probes,final_memory_probes,probed_context,final_probed_context
    context_num = ["2","3","4","5","6"]
    for i,t in enumerate([30, 70, 110, 150, 190]):
        for added_trials in range(0,10):
            response_sorted.append(response_check[t+added_trials])
            memory_probes.append(valid_probe_images[t+added_trials])
            probed_context.append(context_num[i])
    for i,probe in enumerate(memory_probes):
        if response_sorted[i]==1:
            final_memory_probes.append(probe)
            final_probed_context.append(probed_context[i])
    invalid_idx = 0
    while final_memory_probes != 60:
        final_memory_probes.append(invalid_probe_images[invalid_idx]) # Add 10 random new shuffled imgs and then any leftover if trials were skipped
        final_probed_context.append("NA") # Not tested
        invalid_idx +=1
        if invalid_idx == 24: # In case they need more than 15 extra images (should not happen)
            break

    mem_idx = list(range(len(final_memory_probes)))
    random.shuffle(mem_idx)

    final_memory_probes = [final_memory_probes[i] for i in mem_idx]
    final_probed_context = [final_probed_context[i] for i in mem_idx]
init_responses()
print(final_probed_context)
print(len(payout[0]))