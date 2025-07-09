from psychopy import visual, core, event, data, sound,monitors
import random
import numpy as np
import csv
import sys 
import os
import yaml
import random
# Load config
with open("study.yaml", "r") as f:
    config = yaml.safe_load(f)

# Image prefix
image_prefix = config["image_prefix"]

# Initialize the window
# For scanner
monitor = monitors.Monitor("expMonitor", width=config['params']['SCREENWIDTH'])
monitor.setSizePix((config['params']['HRES'], config['params']['VRES']))
monitor.saveMon()

# win = visual.Window([config['params']['HRES'], config['params']['VRES']], allowGUI=True, monitor=monitor, units="norm", color="white", fullscr=True, screen=0)
win = visual.Window(fullscr=True, color='white',allowStencil=True,units = "norm")
experiment_clock = core.Clock()
island_clock = core.Clock()
# Debug and block length
debug = config.get("debug", False)
block_length = config["block_length_debug"] if debug else config["block_length"]

# Bonus money init
bonus_money = 0
bonus_correct = 0

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

source_question = image_prefix + "miscellaneous/source_question.png"
example_probe = image_prefix + "tutorial/example_probe.png"
contingency = image_prefix + "tutorial/contingency.png"

# Practice Images
deck = image_prefix + "miscellaneous/deck.png"
ahoy = image_prefix + "travel/ahoy.png"
desert_welcome_text = image_prefix + "travel/welcome_desert.png"
desert_img = image_prefix + "contexts/context_desert.png"
cavern_welcome_text = image_prefix + "travel/welcome_cavern.png"
cavern_img = image_prefix + "contexts/context_cavern.png"
remember_text_img = image_prefix + "miscellaneous/remember.png"
all_pirates = image_prefix + "pirates/pirates_all.png"
red_pirate = image_prefix + "pirates/red_beard.png"
white_pirate = image_prefix + "pirates/white_beard.png"
black_pirate = image_prefix + "pirates/black_beard.png"

pick_pirate_tutorial = image_prefix + "tutorial/example_pick_best.png"
pt2_source_memory_img = image_prefix + "tutorial/example_source_memory.png"

reward = image_prefix + "rewards/reward.png"
no_reward = image_prefix + "rewards/reward_no.png"

probe_ship = image_prefix + "miscellaneous/cargo_ship.png"
bye_island = image_prefix + "travel/bye.png"

best_pirate = image_prefix + "tutorial/prac_best_pirate.png"
pt2_all_pirates = image_prefix + "tutorial/pirates_all_crop.png"
source_practice = image_prefix + "tutorial/prac_source.png"
incorrect_quiz_feedback = ["That’s incorrect. You win bonus money by collecting gold coins.",
                           "That’s incorrect. Which pirate is the best may change during your time on the island.",
                           "That’s incorrect. How good a pirate is at robbing ships will change from island to island!",
                           "That’s incorrect. You should remember the island where a pirate robbed a ship. To help you remember, you can make up a story associating the object on the ship with the island."
                           ]
# Practice Probes
practice_probes = [
  "probes/probes-256.png",
  "probes/probes-257.png",
  "probes/probes-258.png",
  "probes/probes-259.png",
  "probes/probes-260.png"]

practice_probes_cavern = [
  "probes/probes-261.png",
  "probes/probes-262.png",
  "probes/probes-263.png",
  "probes/probes-264.png",
  "probes/probes-265.png"
]

response_check = []

# Stacked images
desert_welcome = [deck,desert_img,ahoy,desert_welcome_text]
desert_pirates = [deck,desert_img,all_pirates]
desert_red = [deck,desert_img,red_pirate]
desert_white = [deck,desert_img,white_pirate]
desert_black = [deck,desert_img,black_pirate]
desert_bye = [deck,desert_img,bye_island]

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
cavern_bye = [deck,cavern_img,bye_island]

cavern_red_remember = []
cavern_red_reward = []
cavern_white_remember = []
cavern_white_reward = []
cavern_black_remember = []
cavern_black_reward = []

practice_rewards = {
    1: [reward, reward, reward, no_reward, reward],
    2: [no_reward, reward, no_reward, reward, no_reward],
    3: [no_reward, no_reward, reward, no_reward, no_reward]
}
practice_rewards_cavern = {
    1: [reward, reward, reward, no_reward, reward],
    2: [no_reward, no_reward, reward, no_reward, no_reward],
    3: [no_reward, reward, no_reward, reward, reward]
}

for i,probe in enumerate(practice_probes):
    desert_red_remember.append([deck,desert_img,red_pirate,probe_ship,image_prefix+probe,remember_text_img])
    desert_red_reward.append([deck,desert_img,red_pirate,probe_ship,image_prefix+probe,remember_text_img,practice_rewards[1][i]])
    desert_white_remember.append([deck,desert_img,white_pirate,probe_ship,image_prefix+probe,remember_text_img])
    desert_white_reward.append([deck,desert_img,white_pirate,probe_ship,image_prefix+probe,remember_text_img,practice_rewards[2][i]])
    desert_black_remember.append([deck,desert_img,black_pirate,probe_ship,image_prefix+probe,remember_text_img])
    desert_black_reward.append([deck,desert_img,black_pirate,probe_ship,image_prefix+probe,remember_text_img,practice_rewards[3][i]])

for i,probe in enumerate(practice_probes_cavern):
    cavern_red_remember.append([deck,cavern_img,red_pirate,probe_ship,image_prefix+probe,remember_text_img])
    cavern_red_reward.append([deck,cavern_img,red_pirate,probe_ship,image_prefix+probe,remember_text_img,practice_rewards_cavern[1][i]])
    cavern_white_remember.append([deck,cavern_img,white_pirate,probe_ship,image_prefix+probe,remember_text_img])
    cavern_white_reward.append([deck,cavern_img,white_pirate,probe_ship,image_prefix+probe,remember_text_img,practice_rewards_cavern[2][i]])
    cavern_black_remember.append([deck,cavern_img,black_pirate,probe_ship,image_prefix+probe,remember_text_img])
    cavern_black_reward.append([deck,cavern_img,black_pirate,probe_ship,image_prefix+probe,remember_text_img,practice_rewards_cavern[3][i]])

# Init trials
curr_trial = 0
curr_prac_trial = 0
probed_mem_trial = 0
response_sorted = []
memory_probes = []
final_memory_probes = []
probed_context = []
final_probed_context = []
old_probe_list = []
new_probe_list = []
stacked_recog = []

# Instruction init
space_bar = "\n\n[Press the space bar to continue]"
welcome_txt = "Welcome! This is the first part of the study. It will last ~30 minutes. In part 1 and part 2 of the study (total 1 hour), you are the head captain of a pirate ship traveling around the world to different islands. You will play a few different games throughout these two parts. Here is an overview of them:\n\nPart 1 \n\n1. Instructions, practice game, and quiz\n\n2. Pick a pirate game on 6 different islands\n\nPart 2 \n\n3. Pick a pirate game and memory game\n\n4. Where did you see this ship? " + space_bar
different_places = "Today, you’ll visit these 6 islands on your journey. You'll be able to see these islands from your ship."+ space_bar
goal_of_game_1 = " The pirates on your ship will rob other ships as they leave the island. These ships have just sold their goods to the islanders, so they will be filled with lots and lots of gold" + space_bar

goal_of_game_2a = "As the head captain, you do not have to rob any ships yourself. Another pirate will be doing the robbing for you. First, you will choose a pirate to rob the next ship."
goal_of_game_2b = "Then, you'll see if the pirate successfully robbed the ship of their gold. If they were successful, then you'll get a stack of gold coins like this:"
goal_of_game_2c = "If they were not successful, then you'll get no gold coins, and you'll see a big red x like this:"
goal_of_game_2d = "How much bonus money you make is based on how many gold coins you collect." + space_bar

probabilistic = "Even the most skilled pirate can not rob EVERY ship.\nSome ships will have very strong protections against pirate attacks.\nBlue beard, here, is very, very good at robbing ships, but he won't be successful every time he tries to rob one.\nYou can press the ‘1’ key on the keyboard to choose him.\nTry choosing him 10 times to see how often he succeeds at robbing a ship."
blue_beard_outcome = " See, he succeeded in robbing the ships most of the time but not every time. "+ space_bar
pick_pirate = "From these three pirates, you will get to choose which one you want to rob the next ship.\n\nPress the ‘1’ key on your keyboard to pick the pirate with the red beard.\nPress ‘2’ to pick the pirate with the white beard.\nPress ‘3’ to pick the pirate with the black beard.\n\nTry picking a pirate now!"
red_won = "Yay! This pirate succeeded in robbing the ship!"+ space_bar
white_won = "Yay! This pirate succeeded in robbing the ship!"+ space_bar
black_won = "Yay! This pirate succeeded in robbing the ship!"+ space_bar
pick_pirate_again = "Now try choosing another."
red_loss = "Oh no! This pirate did not succeed in robbing the ship."+ space_bar
white_loss = "Oh no! This pirate did not succeed in robbing the ship."+ space_bar
black_loss = "Oh no! This pirate did not succeed in robbing the ship."+ space_bar
time_out = "If you don’t make your choice fast enough, you’ll have to wait a few seconds before you can make another one."+ space_bar
probe = "Once you've chosen a pirate, you’ll be shown the ship they are robbing. You will never rob the same ship twice. Ships can be told apart from one another by the image on them. These are the goods that they sell. See, this ship has a travel mug on it.\n\nYou’ll have to remember which island you saw each ship on. You will win more gold coins and hence more bonus money if you remember correctly! To help you remember, you can imagine a story. For example, here, you could imagine a mug full of water in the desert. Or, as another example, if on the forest island, your pirate robbed a ship with an apple on it, you could imagine an apple falling from a tree in the forest."+ space_bar
changepoint = "How successful a pirate is at robbing ships will depend on the island you’re on. A pirate may have visited this island many times before and gained a lot of practice robbing ships there.\n\nSo, they’re more likely to be successful than a pirate who has never visited the island before."+ space_bar
drift = "How successful a pirate is at robbing ships can also change over the time spent on the island.\nShips may hear from islanders about the pirates coming and will improve their protections against the attack. This may make it harder to rob them.\nShips may also become lazy and weaken the strength of their protections. This may make it easier to rob them.\n\nThings are always changing on the high seas! So, try your best to pay attention!"+ space_bar
summary = "Let's go over the instructions quickly again. You have two important things to do:\n\n 1. Pick the pirate who is the best at robbing ships on the current island. \n\n2. Remember on which island a ship was robbed. The amount of bonus money you can win depends on both.\n\n\nLet’s try a practice game. The game will start by showing you the pirates. First, pick a pirate using the 1, 2, 3 keys on your keyboard. When you are shown a ship, try to remember which island you’re on by making up a story.\n\nThis is just a practice game, so you’re not playing for money.\n\nGood luck! This game will be very difficult but try your best!"+ space_bar
quiz_intro = "Good job on the practice game! Now, you will be asked some true or false questions to make sure you really understand the rules of the game. Press '1' on the keyboard for true and press '2' for false."+ space_bar
q1 = "You win bonus money by collecting gold coins.\n\n\n\nPress 1 for true and press 2 for false."
q2 = "The pirate who is the best at robbing ships when you first arrive on the island will definitely still be the best when you leave the island.\n\n\n\nPress 1 for true and press 2 for false."
q3 = "The pirate who is the best at robbing ships on the first island will be the best on every other island.\n\n\n\nPress 1 for true and press 2 for false."
q4 = "You should remember the island where a pirate robbed a ship.\n\n\n\nPress 1 for true and press 2 for false."

#/////////////////////////////////////////////////////////////////////////
#images day2
example_recog = image_prefix + "tutorial/example_recog_trial.png"
memory_reward = image_prefix + "tutorial/reward_small.png"
# text
goal_summary = "In this part, like before, you get to pick which pirate you want to rob the next ship.\n\nIf the pirate is successful in robbing a ship, you get gold coins that look like this:"
goal_summary2 = "If they were not successful in robbing the ship, then you will get no gold coins and you'll see a big red x like this:"
how_to_pick_summary = "Also, just like before, you will use the 1,2,3 keys on your keyboard to pick a pirate. Press '1' to choose red beard, '2' to choose white beard, and '3' to choose black beard. You only have 3 seconds to pick a pirate, so please make a choice quickly!" + space_bar

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

final_place = "You’ve arrived at the final island of your journey! Unfortunately, it’s very foggy out, so you won’t be able to see it in the distance like the other islands you've visited.\n\nYou’ll still be deciding which pirate you want to rob a ship like before. However, because of the fog, you also won’t be able to see the ship.\n\nThis means you do not have to remember the ships you rob on this island.\n\nUnlike before, you won't be visiting multiple islands today. You will stay on this island for the entire game. This part will last ~30 minutes." + space_bar
recognition_1 = "Sometimes you’ll be shown a ship, and you will be asked if you saw this ship on a past island. Here is an example of what you will see." + space_bar
recognition_2 = "You will press the 5, 6, 7, 8 keys on your keyboard to respond.\n\nIf you are sure you saw the ship, press 5 on your keyboard.\n\nIf you think you saw it before but aren't sure, then press 6.\n\nIf you think you have not seen it before but aren't sure, then press 7.\n\nIf you are sure you did not see the ship before, then press 8." + space_bar
recognition_3 = " If you remember correctly, then you’ll get a gold coin like this:"
recognition_3b = "If you do not remember correctly, then you'll see a red x like this:"
recognition_3c = "Importantly, when you are shown a ship, take the time to remind yourself which island you saw it on. This will help you in the next part of the study." + space_bar
begin_final = "Let's go over the important points again. Your job is to:\n\n\n\n1. Pick the pirate who is best at robbing ships and will bring you back gold coins.\n\n2. Correctly remember whether or not you saw a ship before. If you did see the ship before, remind yourself on which island you saw it.\n\n"+space_bar

source_memory = "Ok, you’re almost done! In this part, you’ll see a ship you saw yesterday, and you will have to pick the island on which you saw it using the keys 1, 2, 3, 4, 5, 6 on your keyboard. The number above each picture tells you which key to press to pick that island. Every time you pick the right island you’ll win some more bonus money, so try your best to remember!"
pick_best_pirate = " Ok, this is your final game of the day. You’ll be shown an island, and you’ll have to pick the pirate you thought was the best at robbing ships on that island. You will use the 1, 2, 3 keys on your keyboard. Just like before, press '1' to choose red beard, '2' to choose white beard, and '3' to choose black beard. Once you pick a pirate, a gold box will surround your choice. Then, you’ll have to pick the pirate you thought was the second best at robbing ships on that island. Once you pick a pirate, a silver box will surround your choice. Then, you’ll move on to the next island."
#/////////////////////////////////////////////////////////////////////////



# Reward
import numpy as np

# ----- Constants -----
num_bandits = config['params']['num_bandits']
first_block = config['params']['first_block']
block_len = config['params']['block_len']
num_blocks = config['params']['num_blocks']
num_trials = block_len * num_blocks + 130 # 10 trials skipped
init_payoff = config['params']['init_payoff']
decayTheta = init_payoff.copy()
payoff_bounds = config['params']['payoff_bounds']
decay_lambda = config['params']['decay_lambda']
drift_noise = config['params']['drift_noise']
rotation_trials = config['params']['rotation_trials']
deterministic_trials = config['params']['deterministic_trials']
ctx_bump = config['params']['ctx_bump']

# ------------------------- BELOW IS OLD CODE NEED TO REMOVE ------------------------------
# Add bumped deterministic trials

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

probabilities = {
    1: payout[0] * 0.01,
    2: payout[1] * 0.01,
    3: payout[2] * 0.01
}
# ------------------------- ABOVE IS OLD CODE NEED TO REMOVE ------------------------------

import pandas as pd

# Load the CSV file
payoutNum = np.random.randint(1,5)
df = pd.read_csv(f'../run_exp/static/original_payouts/dmt-0{payoutNum}.csv',header=None)  # Replace with your actual path

# Convert to probability dict
probabilities = {
    1: df[0].to_numpy(),
    2: df[1].to_numpy(),
    3: df[2].to_numpy()
}

# Not needed but in case needed
payout = {
    1: probabilities[1] / 0.01,
    2: probabilities[2] / 0.01,
    3: probabilities[3] / 0.01
}

ifReward = {key: np.random.binomial(1, prob) for key, prob in probabilities.items()}

reward_imgs = {
    key: np.where(value == 1, reward, no_reward)
    for key, value in ifReward.items()
}

# Valid probes (seen in task) and Invalid (foils during memory phase)
valid_probe_images = []
invalid_probe_images = []
for i in range(1,181): # 180 trials (30 trials * 6 blocks)
    if i < 10:   
        valid_probe_images.append(image_prefix + f"probes/probes-0{i}.png")
    else:
        valid_probe_images.append(image_prefix + f"probes/probes-{i}.png")
for i in range(231,256): # 25 trials (wrong probe trials during testing)
    invalid_probe_images.append(image_prefix + f"probes/probes-{i}.png")

random.shuffle(valid_probe_images)
random.shuffle(invalid_probe_images)

sample_window = 10
mean_ct = 5
min_ct = 1
max_ct = 8
num_probes = sample_window * (num_blocks-1)
num_invalid_probes = np.round(num_probes/5)
num_probes = int(num_probes + num_invalid_probes)
second_half_trials = np.arange(190, 310)  # 120 trials
available_for_mem_probe = []

for b in range(0, num_blocks):  
    block_start = block_len * b
    block_candidates = list(range(block_start, block_start + sample_window))
    available_for_mem_probe.extend(block_candidates)

# Group by block
from collections import defaultdict
block_to_candidates = defaultdict(list)

for trial in available_for_mem_probe:
    block = trial // block_len
    block_to_candidates[block].append(trial)

# Sample ~8-9 from each block
valid_probe_trials = []
for block in sorted(block_to_candidates.keys()):
    trials_in_block = block_to_candidates[block]
    n_to_sample = 9 if block < 5 else 5  # e.g., 5 from last block to make total = 50
    sampled = np.random.choice(trials_in_block, size=n_to_sample, replace=False).tolist()
    valid_probe_trials.extend(sampled)

num_invalid_probes = 10
num_total_probes = len(valid_probe_trials) + num_invalid_probes  # = 60

log_rand = np.log(np.random.rand(num_total_probes))
log_rand_div_mean_ct = log_rand / (1 / mean_ct)
choice_blocks = np.ceil(log_rand_div_mean_ct) * -1 + min_ct
choice_blocks = np.clip(choice_blocks, min_ct, max_ct).astype(int)

# Adjust to make sum exactly num of pt 2 trials (120)
while (choice_blocks.sum() != 120 or
       np.any(choice_blocks > max_ct)):

    ind = np.random.randint(0, num_total_probes)
    delta = int(np.sign(choice_blocks.sum() - (120)))
    choice_blocks[ind] -= delta
    choice_blocks = np.clip(choice_blocks, min_ct, max_ct)

choice_blocks[0] += 10

# # Step 4: Compute memory probe trial indices
# block_sizes = choice_blocks + 1
# cumsum_blocks = np.cumsum(block_sizes)
# mem_probe_trials = cumsum_blocks + (190)  # shift into 2nd half
# mem_probe_trials = mem_probe_trials - 1  # zero-indexing

# # Step 5: Compute choice trials (set difference)
# all_trials = np.arange(num_trials)
# choice_trials = np.setdiff1d(all_trials, mem_probe_trials)
# Context-based trial order

contexts = [image_prefix + "contexts/context_coral_beach.png",
            image_prefix + "contexts/context_sunleaf_forest.png",
            image_prefix + "contexts/context_icecap_mountain.png",
            image_prefix + "contexts/context_driftwood_beach.png",
            image_prefix + "contexts/context_stonepine_forest.png",
            image_prefix + "contexts/context_greenrock_mountain.png"]

welcomeArray = [image_prefix + "travel/welcome_coral_beach.png",
                image_prefix + "travel/welcome_sunleaf_forest.png",
                image_prefix + "travel/welcome_icecap_mountain.png",
                image_prefix + "travel/welcome_driftwood_beach.png",
                image_prefix + "travel/welcome_stonepine_forest.png",
                image_prefix + "travel/welcome_greenrock_mountain.png"]

# Init main phase image stacks
stacked_all_pirates = []
stacked_red_remember = []
stacked_red_reward = []
stacked_white_remember = []
stacked_white_reward = []
stacked_black_remember = []
stacked_black_reward = []
context_labels = []
stacked_planet_welcome = []
stacked_red_pirate = []
stacked_white_pirate = []
stacked_black_pirate = []
stacked_island_bye = []
stacked_island_nopirate = []
prior_trial = 0
trial_range = 30
for context_idx,context in enumerate(contexts):
    for trial_idx in range(prior_trial,trial_range): # Rest of the contexts are 40 trials
        stacked_planet_welcome.append([deck,context,ahoy,welcomeArray[context_idx]])
        stacked_island_nopirate.append([deck,context])
        stacked_all_pirates.append([deck,context,all_pirates])
        stacked_red_pirate.append([deck,context,red_pirate])
        stacked_red_remember.append([deck,context,red_pirate,probe_ship,valid_probe_images[trial_idx]])
        stacked_red_reward.append([deck,context,red_pirate,probe_ship,valid_probe_images[trial_idx],reward_imgs[1][trial_idx]])
        stacked_white_pirate.append([deck,context,white_pirate])
        stacked_white_remember.append([deck,context,white_pirate,probe_ship,valid_probe_images[trial_idx]])
        stacked_white_reward.append([deck,context,white_pirate,probe_ship,valid_probe_images[trial_idx],reward_imgs[2][trial_idx]])
        stacked_black_pirate.append([deck,context,black_pirate])
        stacked_black_remember.append([deck,context,black_pirate,probe_ship,valid_probe_images[trial_idx]])
        stacked_black_reward.append([deck,context,black_pirate,probe_ship,valid_probe_images[trial_idx],reward_imgs[3][trial_idx]])
        stacked_island_bye.append([deck,context,bye_island])
        context_labels.append(context.split("contexts/context_")[-1].split(".png")[0])
    prior_trial += 30
    trial_range += 30

# Part 2 stacks

stacked_seven_room = []
context_seven = image_prefix + "contexts/context_blank.png"

stacked_seven_red = []
stacked_seven_white = []
stacked_seven_black = []

stacked_seven_red_reward = []
stacked_seven_white_reward = []
stacked_seven_black_reward = []
stacked_seven_room_pirates = []

source_memory_contexts = [image_prefix + "contexts/source_1_beach.png",image_prefix + "contexts/source_2_forest.png",image_prefix + "contexts/source_3_mountain.png",
                          image_prefix + "contexts/source_4_beach.png",image_prefix + "contexts/source_5_forest.png",image_prefix + "contexts/source_6_mountain.png"]

pt2_index = 0

# Seventh Room
for i in range(180,num_trials):
    stacked_seven_room.append([deck,context_seven])
    stacked_seven_room_pirates.append([deck,context_seven,all_pirates])
    stacked_seven_red.append([deck,context_seven,red_pirate])
    stacked_seven_white.append([deck,context_seven,white_pirate])
    stacked_seven_black.append([deck,context_seven,black_pirate])
    stacked_seven_red_reward.append([deck,context_seven,red_pirate,reward_imgs[1][i]])
    stacked_seven_white_reward.append([deck,context_seven,white_pirate,reward_imgs[2][i]])
    stacked_seven_black_reward.append([deck,context_seven,black_pirate,reward_imgs[3][i]])

recog_question = image_prefix + "miscellaneous/probe_recog.png"

# Best Pirate
best_question = image_prefix + "pick_best/best_question.png"

red_best = image_prefix + "pick_best/red_best.png"
white_best = image_prefix + "pick_best/white_best.png"
black_best = image_prefix + "pick_best/black_best.png"

stacked_best_pirate = []
context_order_labels = []
for c in contexts:
    stacked_best_pirate.append([best_question,c,all_pirates])
    context_order_labels.append(c.split("contexts/context_")[-1].split(".png")[0])
best_pirate_trial = 0


# Sequences
ship_sequence = [image_prefix + r for r in config["ship_sequence"]] # For ship travelling between contexts

# Init data collections
block_time = 0
text_index = 0
failedNum = 0

# Init keyboard list from yaml
keyList = [config['params']['BUTTON_1'],config['params']['BUTTON_2'],config['params']['BUTTON_3']]
recogKeyList = [config['params']['BUTTON_5'],config['params']['BUTTON_6'],config['params']['BUTTON_7'],config['params']['BUTTON_8']]
source_key_list = [config['params']['BUTTON_1'],config['params']['BUTTON_2'],config['params']['BUTTON_3'],config['params']['BUTTON_4'],config['params']['BUTTON_5'],config['params']['BUTTON_6']]
if len(sys.argv) < 2: # Call participant ID in terminal
    print("Error: No participant ID provided.")
    print("Usage: python3 experiment.py <participant_id>")
    sys.exit(1)  # Exit with error code 1

# Get participant ID from command-line argument
participant_id = sys.argv[1]

# Init data var
study = []

# For study.yaml output
def study_filename(subject_id_str):
    extension = ".yaml"
    if not subject_id_str:
        return 'study' + extension
    return 'study' + "." + str(subject_id_str) + extension

# Updates to study.yaml
def write_study():
    """Writes study.yaml file with data output var"""
    if not participant_id:
        raise Exception("Shouldn't write to original study file! Please provide a valid subject ID.")
    with open(study_filename(participant_id), 'w') as file:
        for row in study:
            if isinstance(row.get("AlienOrder"), np.ndarray):
                row["AlienOrder"] = row["AlienOrder"].tolist()
        yaml.dump(study, file)

# Modify probe numbers for data collection
probe_nums = []
for valid_img_probe_num in valid_probe_images:
    probe_nums.append(valid_img_probe_num.split("images/probes/probes-")[-1].split(".png")[0])


# Initial data format -- What the csv reader takes as column names
study.append({
    "ID": participant_id,
    "TrialType":f"InitializeStudy",
    "PayoutDistNum":payoutNum,
    "BlockNum": "",
    "contextOrder": context_labels, # List of context orders
    "reward_rate_red":probabilities[1].tolist(), # List of reward rates for red pirate and below is white and black
    "reward_rate_white":probabilities[2].tolist(),
    "reward_rate_black":probabilities[3].tolist(),
    "probe_order": probe_nums,
    "QuizFailedNum": "",
    "TimeElapsed": experiment_clock.getTime(),
    "key_press": "",
    "RT": "",
    "context": "",
    "reward_prob_red":"",
    "reward_prob_white":"",
    "reward_prob_black":"",
    "choice":"",
    "probe":"",
    "reward":"",
    "confidence":"",
    "TimeInBlock": "",
    "Bonus":""
})
# init in case exp breaks
write_study()  

def show_text(text, image_path=None,x=1,y=1,height=0,img_pos = -0.3,text_height=config['params']['FONT_SIZE'],keys=['space'],duration=0):
    """Displays a text message either until a key is pressed or for a specified duration (Default unlimited duration)"""
    global text_index,study,experiment_clock
    text_index += 1 # Used for trial data descriptions

    # If there is an image (specified by image_path) it will show it on the screen
    if image_path:
        stim_image = visual.ImageStim(win, image=image_path, size=(x, y), pos=(0, img_pos))  # Adjust size as needed
        stim_image.draw()

    stim = visual.TextStim(win, text=text, color='black', height=text_height, pos=(0, height), wrapWidth=config['params']['TEXTBOX_WIDTH']) # Adds whatever text is called
    stim.draw() # Pushes it to screen

    win.flip() # Resets screen
    # Only keys taken in exp (need to change to specify which key to use)
    if duration > 0:
        timer = core.Clock()
        while timer.getTime() < duration:
            keys = event.getKeys(keyList=["space"])
            if "space" in keys:
                break 
    else:
        event.waitKeys(keyList=keys)
    study.append({
        "ID": "",
        "TrialType":f"Instruction_{text_index}",
        "PayoutDistNum":"",
        "BlockNum": "",
        "contextOrder": "",
        "reward_rate_red":"",
        "reward_rate_white":"",
        "reward_rate_black":"",
        "probe_order": "",
        "QuizFailedNum": "",
        "TimeElapsed": experiment_clock.getTime(),
        "key_press": "",
        "RT": "",
        "context": "",
        "reward_prob_red":"",
        "reward_prob_white":"",
        "reward_prob_black":"",
        "choice":"",
        "probe":"",
        "reward":"",
        "confidence":"",
        "TimeInBlock": "",
        "Bonus":""
    })

def show_multi_img_text(texts=[], image_paths=None,x=1,y=1,heights=[],img_pos=[],text_height=config['params']['FONT_SIZE'],keys=['space']):
    """Displays a text message either until a key is pressed or for a specified duration (Default unlimited duration)"""
    global text_index,study,experiment_clock
    text_index += 1 # Used for trial data descriptions
    
    for i,text in enumerate(texts):
        stim = visual.TextStim(win, text=text, color='black', height=text_height, pos=(0, heights[i]), wrapWidth=config['params']['TEXTBOX_WIDTH']) # Adds whatever text is called
        stim.draw() # Pushes it to screen

    # If there is an image (specified by image_path) it will show it on the screen
    if image_paths:
        for j,image in enumerate(image_paths):
            stim_image = visual.ImageStim(win, image=image, size=(x, y), pos=(0, img_pos[j]))  # Adjust size as needed
            stim_image.draw()

    win.flip() # Resets screen
    event.waitKeys(keyList=keys)

    study.append({
        "ID": "",
        "TrialType":f"Instruction_{text_index}",
        "PayoutDistNum":"",
        "BlockNum": "",
        "contextOrder": "",
        "reward_rate_red":"",
        "reward_rate_white":"",
        "reward_rate_black":"",
        "probe_order": "",
        "QuizFailedNum": "",
        "TimeElapsed": experiment_clock.getTime(),
        "key_press": "",
        "RT": "",
        "context": "",
        "reward_prob_red":"",
        "reward_prob_white":"",
        "reward_prob_black":"",
        "choice":"",
        "probe":"",
        "reward":"",
        "confidence":"",
        "TimeInBlock": "",
        "Bonus":""
    })
# Shows an image for some duration
def show_image(img_path, duration=1.5):
    """Displays an image for a given duration"""
    stim = visual.ImageStim(win, image=img_path,size=(1.2,1.2))
    stim.draw()
    win.flip()
    core.wait(duration)

def show_stacked_images(img_paths = [], duration=3,y=None):
    """Displays an image for a given duration"""
    if y:
        for i,img_path in enumerate(img_paths):
            stim = visual.ImageStim(win, image=img_path,size=(1.2,1.2),pos=(0,y[i]))
            stim.draw()
        win.flip()
    else:
        for img_path in img_paths:
            stim = visual.ImageStim(win, image=img_path,size=(1.2,1.2),pos=(0,0))
            stim.draw()
        win.flip()
    core.wait(duration)

# Timeout image for 2 seconds
def too_slow():
    show_image(timeout_img,duration=2)
    study.append({
        "ID": "",
        "TrialType":f"time_out",
        "PayoutDistNum":"",
        "BlockNum": "",
        "contextOrder": "",
        "reward_rate_red":"",
        "reward_rate_white":"",
        "reward_rate_black":"",
        "probe_order": "",
        "QuizFailedNum": "",
        "TimeElapsed": experiment_clock.getTime(),
        "key_press": "",
        "RT": "",
        "context": "",
        "reward_prob_red":"",
        "reward_prob_white":"",
        "reward_prob_black":"",
        "choice":"",
        "probe":"",
        "reward":"",
        "confidence":"",
        "TimeInBlock": "",
        "Bonus":""
    })

# Rocket travel trials
def travel_trial():
    """Simulates travel sequence"""
    global study
    for img in ship_sequence: # Animates through all images and then waits 1 sec after for 10 sec total
        stim = visual.ImageStim(win, image=img, size=(1.2,1.2))
        stim.draw()
        win.flip()
        core.wait(1)
    core.wait(1)
    study.append({
        "ID": "",
        "TrialType":f"new_island",
        "PayoutDistNum":"",
        "BlockNum": "",
        "contextOrder": "",
        "reward_rate_red":"",
        "reward_rate_white":"",
        "reward_rate_black":"",
        "probe_order": "",
        "QuizFailedNum": "",
        "TimeElapsed": experiment_clock.getTime(),
        "key_press": "",
        "RT": "",
        "context": "",
        "reward_prob_red":"",
        "reward_prob_white":"",
        "reward_prob_black":"",
        "choice":"",
        "probe":"",
        "reward":"",
        "confidence":"",
        "TimeInBlock": "",
        "Bonus":""
    })
    win.flip()

# For the practice instructions
def practice_blue_loop():
    """Displays 10 practice trials hard coded to be 3 win, 1 no win, 3 win, 1 no win, 2 win"""
    for i in range(0,3):
        show_image(img_path=blue_win,duration=1.5)
        show_text("Press the '1' key on the keyboard to pick blue beard.",image_path=tutorial_blue_pirate,x=1.2,y=1.2,height=0.0,text_height=0.05,keys=['1'],img_pos=0.0)
    show_image(img_path=blue_no_win,duration=1.5)
    show_text("Press the '1' key on the keyboard to pick blue beard.",image_path=tutorial_blue_pirate,x=1.2,y=1.2,height=0.0,text_height=0.05,keys=['1'],img_pos=0.0)
    for i in range(0,3):
        show_image(img_path=blue_win,duration=1.5)
        show_text("Press the '1' key on the keyboard to pick blue beard.",image_path=tutorial_blue_pirate,x=1.2,y=1.2,height=0.0,text_height=0.05,keys=['1'],img_pos=0.0)
    show_image(img_path=blue_no_win,duration=1.5)
    show_text("Press the '1' key on the keyboard to pick blue beard.",image_path=tutorial_blue_pirate,x=1.2,y=1.2,height=0.0,text_height=0.05,keys=['1'],img_pos=0.0)
    for i in range(0,2):
        show_image(img_path=blue_win,duration=1.5)
        if i != 1:
            show_text("Press the '1' key on the keyboard to pick blue beard.",image_path=tutorial_blue_pirate,x=1.2,y=1.2,height=0.0,text_height=0.05,keys=['1'],img_pos=0.0)

# For the practice quiz
def run_quiz():
    """Displays a set of true false quiz questions and checks answers."""
    global failedNum
    questions = [q1,q2,q3,q4]
    correct = [True,False,False,True] # Is the first option the correct one
    allCorrect = True # To check if they get one wrong
    for i,question in enumerate(questions):
        stim = visual.TextStim(win, text=question, color='black', height=config['params']['FONT_SIZE'], pos=(0, 0), wrapWidth=config['params']['TEXTBOX_WIDTH']) # Adds whatever text is called
        stim.draw() # Pushes it to screen
        win.flip() # Resets screen
        # Only keys taken in exp (need to change to specify which key to use)
        keys = event.waitKeys(keyList=[keyList[0],keyList[1]])
        if keys:
            key = keys[0]
            if correct[i]:
                if keyList[0] in key:
                    show_text("That's correct!" + space_bar)
                elif keyList[1] in key:
                    show_text(incorrect_quiz_feedback[i] + space_bar)
                    allCorrect= False
            else:
                if keyList[0] in key:
                    show_text(incorrect_quiz_feedback[i] + space_bar)
                    allCorrect= False
                elif keyList[1] in key:
                    show_text("That's correct!" + space_bar)
    if allCorrect:
        study.append({
            "ID": "",
            "TrialType":f"quiz_complete",
            "PayoutDistNum":"",
            "BlockNum": "",
            "contextOrder": "",
            "reward_rate_red":"",
            "reward_rate_white":"",
            "reward_rate_black":"",
            "probe_order": "",
            "QuizFailedNum": failedNum,
            "TimeElapsed": experiment_clock.getTime(),
            "key_press": "",
            "RT": "",
            "context": "",
            "reward_prob_red":"",
            "reward_prob_white":"",
            "reward_prob_black":"",
            "choice":"",
            "probe":"",
            "reward":"",
            "confidence":"",
            "TimeInBlock": "",
            "Bonus":""
        })
        pass
    else:
        show_text("Oops, you missed some questions. Now that you’ve heard the correct answers. Try the quiz again!" + space_bar)
        failedNum +=1
        run_quiz()


def practice_pirates(text=pick_pirate,switch='win'):
    global study,experiment_clock,keyList

    # If there is an image (specified by image_path) it will show it on the screen
    stim_image = visual.ImageStim(win, image=tutorial_all_pirates, size=(1, 1), pos=(0, -0.3))  # Adjust size as needed
    stim_image.draw()

    stim = visual.TextStim(win, text=text, color='black', height=config['params']['FONT_SIZE'], pos=(0,0.6), wrapWidth=config['params']['TEXTBOX_WIDTH']) # Adds whatever text is called
    stim.draw() # Pushes it to screen
    win.flip()
    resp_key = event.waitKeys(keyList=keyList)
    if resp_key:
        key = resp_key[0] # RT used for data collection
        if keyList[0] in key: # 1
            won_text = red_won
            pirate = red_win
            loss_text = red_loss
            pirate_loss=red_no_win
        if keyList[1] in key: # 2
            won_text = white_won
            pirate = white_win
            loss_text = white_loss
            pirate_loss=white_no_win
        if keyList[2] in key: # 3
            won_text = black_won
            pirate = black_win
            loss_text = black_loss
            pirate_loss=black_no_win
    
    if switch == 'win':
        show_text(text=won_text,height=0.6,image_path=pirate)
    elif switch == 'nowin':
        show_text(text=loss_text,height=0.6,image_path=pirate_loss)
            
# Where the main task is run

def practice_pirate_loop(duration=3,setting = 'desert'):
    """For choosing the pirate, getting the probe, and seeing if there is a reward"""
    global curr_prac_trial,study
    if setting == 'desert':
        location = desert_pirates
    elif setting == 'cavern':
        location = cavern_pirates
    for img_path in location:
        stim = visual.ImageStim(win, image=img_path,size=(1.2,1.2))
        stim.draw()
    response_clock = core.Clock()
    win.flip()
    resp_key = event.waitKeys(keyList=keyList,timeStamped=response_clock,maxWait=duration)
    
    if resp_key:
        key,RT = resp_key[0] # RT used for data collection
        if setting == 'desert':
            probeList = practice_probes
            rewards = practice_rewards
            set_img = desert_img
            if keyList[0] in key: # 1
                pirateChoice = desert_red
                pirateProbe = desert_red_remember[curr_prac_trial]
                pirateReward = desert_red_reward[curr_prac_trial]
            if keyList[1] in key: # 2
                pirateChoice = desert_white
                pirateProbe = desert_white_remember[curr_prac_trial]
                pirateReward = desert_white_reward[curr_prac_trial]
            if keyList[2] in key: # 3
                pirateChoice = desert_black
                pirateProbe = desert_black_remember[curr_prac_trial]
                pirateReward = desert_black_reward[curr_prac_trial]
        elif setting == 'cavern':
            probeList = practice_probes_cavern
            set_img = cavern_img
            rewards = practice_rewards_cavern
            if keyList[0] in key: # 1
                pirateChoice = cavern_red
                pirateProbe = cavern_red_remember[curr_prac_trial-5]
                pirateReward = cavern_red_reward[curr_prac_trial-5]
            if keyList[1] in key: # 2
                pirateChoice = cavern_white
                pirateProbe = cavern_white_remember[curr_prac_trial-5]
                pirateReward = cavern_white_reward[curr_prac_trial-5]
            if keyList[2] in key: # 3
                pirateChoice = cavern_black
                pirateProbe = cavern_black_remember[curr_prac_trial-5]
                pirateReward = cavern_black_reward[curr_prac_trial-5]
        show_stacked_images(img_paths=pirateChoice,duration=1)
        show_stacked_images(img_paths=pirateProbe,duration=2)
        show_stacked_images(img_paths=pirateReward,duration=1)
        show_stacked_images(img_paths=[deck,set_img],duration=1)
        if curr_prac_trial < 5:
            trial_num = curr_prac_trial
        else:
            trial_num = curr_prac_trial-5
        study.append({
            "ID": "",
            "TrialType":f"practice_pirate_{curr_prac_trial+1}",
            "PayoutDistNum":"",
            "BlockNum": "",
            "contextOrder": "",
            "reward_rate_red":"",
            "reward_rate_white":"",
            "reward_rate_black":"",
            "probe_order": "",
            "QuizFailedNum": "",
            "TimeElapsed": experiment_clock.getTime(),
            "key_press": key,
            "RT": RT,
            "context": setting,
            "reward_prob_red":"",
            "reward_prob_white":"",
            "reward_prob_black":"",
            "choice":"",
            "probe": probeList[trial_num],
            "reward":rewards[int(key)][trial_num],
            "confidence":"",
            "TimeInBlock": "",
            "Bonus":""
        })
        curr_prac_trial +=1
        if curr_prac_trial < 5:
            practice_pirate_loop()
        elif curr_prac_trial >=5 and curr_prac_trial < 10:
            if curr_prac_trial == 5:
                show_stacked_images(desert_bye)
                travel_trial()
                show_stacked_images(cavern_welcome,duration=3)
            practice_pirate_loop(setting='cavern')
        else:  
            stim = visual.ImageStim(win, image=best_pirate,size=(1.2,1.2))
            stim.draw()
            win.flip()
            best_key = event.waitKeys(keyList=keyList)

            if best_key:
                bestkeys = best_key[0]
                if keyList[0] in bestkeys: # 1
                    show_text("That's correct! Red beard was the best." + space_bar)
                    wrong = 0
                if keyList[1] in bestkeys: # 2
                    show_text("That's incorrect! Red beard was the best."+ space_bar)
                    wrong = 1
                if keyList[2] in bestkeys: # 3
                    show_text("That's incorrect! Red beard was the best."+ space_bar)
                    wrong = 1
            study.append({
                "ID": "",
                "TrialType":f"practce_best_pirate",
                "PayoutDistNum":"",
                "BlockNum": "",
                "contextOrder": "",
                "reward_rate_red":"",
                "reward_rate_white":"",
                "reward_rate_black":"",
                "probe_order": "",
                "QuizFailedNum": wrong,
                "TimeElapsed": experiment_clock.getTime(),
                "key_press": bestkeys,
                "RT": "",
                "context": "",
                "reward_prob_red":"",
                "reward_prob_white":"",
                "reward_prob_black":"",
                "choice":"",
                "probe": "",
                "reward":"",
                "confidence":"",
                "TimeInBlock": "",
                "Bonus":""
            })
            write_study()  
            stim = visual.ImageStim(win, image=source_practice,size=(1.2,1.2))
            stim.draw()
            win.flip()
            source_key = event.waitKeys(keyList=[keyList[0],keyList[1]])

            if source_key:
                sourcekey = source_key[0]
                if keyList[0] in sourcekey: # 1
                    show_text("That's incorrect! You saw this ship on the cavern island."+ space_bar)
                    wrong = 1
                if keyList[1] in sourcekey: # 2
                    show_text("That's correct! You saw this ship on the cavern island." + space_bar)
                    wrong = 0
            study.append({
                "ID": "",
                "TrialType":f"practice_source_memory",
                "PayoutDistNum":"",
                "BlockNum": "",
                "contextOrder": "",
                "reward_rate_red":"",
                "reward_rate_white":"",
                "reward_rate_black":"",
                "probe_order": "",
                "QuizFailedNum": wrong,
                "TimeElapsed": experiment_clock.getTime(),
                "key_press": sourcekey,
                "RT": "",
                "context": "",
                "reward_prob_red":"",
                "reward_prob_white":"",
                "reward_prob_black":"",
                "choice":"",
                "probe": "",
                "reward":"",
                "confidence":"",
                "TimeInBlock": "",
                "Bonus":""
            })
    else:
        show_image(timeout_img,duration=2)
        practice_pirate_loop(setting=setting)
        study.append({
                "ID": "",
                "TrialType":f"time_out",
                "PayoutDistNum":"",
                "BlockNum": "",
                "contextOrder": "",
                "reward_rate_red":"",
                "reward_rate_white":"",
                "reward_rate_black":"",
                "probe_order": "",
                "QuizFailedNum": "",
                "TimeElapsed": experiment_clock.getTime(),
                "key_press": "",
                "RT": "",
                "context": "",
                "reward_prob_red":"",
                "reward_prob_white":"",
                "reward_prob_black":"",
                "choice":"",
                "probe": "",
                "reward":"",
                "confidence":"",
                "TimeInBlock": "",
                "Bonus":""
            })

def learn_phase_loop():
    """For choosing the pirate, getting the probe, and seeing if there is a reward"""
    global curr_trial,study,island_clock
    island_shift_indx = [0,30,60,90,120,180] # 0 index where contexts shift
    if curr_trial in island_shift_indx:
        show_stacked_images(stacked_planet_welcome[curr_trial],duration=3) # Show welcome on first visit
    for img_path in stacked_all_pirates[curr_trial]: # Show all pirates and take responses
        stim = visual.ImageStim(win, image=img_path,size=(1.2,1.2))
        stim.draw()
    response_clock = core.Clock()
    island_clock = core.Clock()
    win.flip()
    resp_key = event.waitKeys(keyList=keyList,timeStamped=response_clock,maxWait=3)
    
    if resp_key:
        key,RT = resp_key[0] # RT used for data collection
        response_check.append(1)
        if keyList[0] in key: # 1
            choice = 'red_pirate'
            pirateChoice = stacked_red_pirate[curr_trial]
            pirateProbe = stacked_red_remember[curr_trial]
            pirateReward = stacked_red_reward[curr_trial]
        if keyList[1] in key: # 2
            choice = 'white_pirate'
            pirateChoice = stacked_white_pirate[curr_trial]
            pirateProbe = stacked_white_remember[curr_trial]
            pirateReward = stacked_white_reward[curr_trial]
        if keyList[2] in key: # 3
            choice = 'black_pirate'
            pirateChoice = stacked_black_pirate[curr_trial]
            pirateProbe = stacked_black_remember[curr_trial]
            pirateReward = stacked_black_reward[curr_trial]
        show_stacked_images(img_paths=pirateChoice,duration=1)
        show_stacked_images(img_paths=pirateProbe,duration=2)
        show_stacked_images(img_paths=pirateReward,duration=1)
        show_stacked_images(img_paths=stacked_island_nopirate[curr_trial],duration=1)
        study.append({
            "ID": "",
            "TrialType":f"pirate_{curr_trial+1}",
            "PayoutDistNum":"",
            "BlockNum": "",
            "contextOrder": "",
            "reward_rate_red":"",
            "reward_rate_white":"",
            "reward_rate_black":"",
            "probe_order": "",
            "QuizFailedNum": "",
            "TimeElapsed": experiment_clock.getTime(),
            "key_press": key,
            "RT": RT,
            "context": context_labels[curr_trial],
            "reward_prob_red":probabilities[1][curr_trial],
            "reward_prob_white":probabilities[2][curr_trial],
            "reward_prob_black":probabilities[3][curr_trial],
            "choice":choice,
            "probe":valid_probe_images[curr_trial],
            "reward":ifReward[int(key)][curr_trial],
            "confidence":"",
            "TimeInBlock": island_clock.getTime(),
            "Bonus":""
        })
        write_study()  
        curr_trial +=1 # Advance trial
        if curr_trial == first_block + (block_len * (num_blocks - 1)):
            show_stacked_images(stacked_island_bye[curr_trial-1],duration=1.5)
            init_responses()
        elif curr_trial in island_shift_indx:
                show_stacked_images(stacked_island_bye[curr_trial-1],duration=1.5)
                take_break()
                travel_trial()
                learn_phase_loop()
        else:
            learn_phase_loop()
    else:
        response_check.append(0)
        show_image(timeout_img,duration=2)
        study.append({
                "ID": "",
                "TrialType":f"time_out",
                "PayoutDistNum":"",
                "BlockNum": "",
                "contextOrder": "",
                "reward_rate_red":"",
                "reward_rate_white":"",
                "reward_rate_black":"",
                "probe_order": "",
                "QuizFailedNum": "",
                "TimeElapsed": experiment_clock.getTime(),
                "key_press": "",
                "RT": "",
                "context": "",
                "reward_prob_red":"",
                "reward_prob_white":"",
                "reward_prob_black":"",
                "choice":"",
                "probe": "",
                "reward":"",
                "confidence":"",
                "TimeInBlock": "",
                "Bonus":""
            })
        learn_phase_loop()

def take_break():
    """Breaks between islands"""
    global island_clock
    show_text(text="Time to take a quick break! You have 2 minutes to rest, but you can move on sooner if you'd like."+space_bar,duration=120)
    study.append({
        "ID": "",
        "TrialType":f"take_break",
        "PayoutDistNum":"",
        "BlockNum": "",
        "contextOrder": "",
        "reward_rate_red":"",
        "reward_rate_white":"",
        "reward_rate_black":"",
        "probe_order": "",
        "QuizFailedNum": "",
        "TimeElapsed": experiment_clock.getTime(),
        "key_press": "",
        "RT": "",
        "context": "",
        "reward_prob_red":"",
        "reward_prob_white":"",
        "reward_prob_black":"",
        "choice":"",
        "probe": "",
        "reward":"",
        "confidence":"",
        "TimeInBlock": island_clock.getTime(),
        "Bonus":""
    })

def init_responses():
    global response_sorted, memory_probes, final_memory_probes, probed_context, final_probed_context
    global old_probe_list, new_probe_list, stacked_recog

    # Reset all lists
    response_sorted = []
    memory_probes = []
    probed_context = []
    final_memory_probes = []
    final_probed_context = []
    old_probe_list = []
    new_probe_list = []
    stacked_recog = []

    # --- PART 1: Select Valid Probes from 1st 10 trials of each block ---
    available_for_mem_probe = []
    for b in range(num_blocks):  
        block_start = block_len * b
        block_candidates = list(range(block_start, block_start + sample_window))
        available_for_mem_probe.extend(block_candidates)

    from collections import defaultdict
    block_to_candidates = defaultdict(list)
    for trial in available_for_mem_probe:
        block = trial // block_len
        block_to_candidates[block].append(trial)

    # Sample ~8-9 trials from each block (sum = 50)
    valid_probe_trials = []
    for block in sorted(block_to_candidates.keys()):
        trials_in_block = block_to_candidates[block]
        n_to_sample = 9 if block < 5 else 5  # 5 from last block
        sampled = np.random.choice(trials_in_block, size=n_to_sample, replace=False).tolist()
        valid_probe_trials.extend(sampled)

    # --- PART 2: Filter valid_probe_trials using response_check ---
    # response_check = np.random.randint(1,2,180)
    # COMMENT OUTTTTT
    context_num = [str(i+1) for i in range(num_blocks)]  # e.g., ['1','2','3','4','5','6']
    for trial_idx in valid_probe_trials:
        response_sorted.append(response_check[trial_idx])
        memory_probes.append(valid_probe_images[trial_idx])
        probed_context.append(context_num[trial_idx // block_len])

    for i, probe in enumerate(memory_probes):
        if response_sorted[i] == 1:
            final_memory_probes.append(probe)
            final_probed_context.append(probed_context[i])

    old_probe_list = final_memory_probes.copy()

    # --- PART 3: Add invalid probes if needed to reach 60 total ---
    invalid_idx = 0
    while len(final_memory_probes) < 60:
        final_memory_probes.append(invalid_probe_images[invalid_idx])
        final_probed_context.append("NA")
        new_probe_list.append(invalid_probe_images[invalid_idx])
        invalid_idx += 1
        if invalid_idx >= len(invalid_probe_images):  # safety check
            break

    # --- PART 4: Shuffle everything ---
    mem_idx = list(range(len(final_memory_probes)))
    random.shuffle(mem_idx)

    final_memory_probes = [final_memory_probes[i] for i in mem_idx]
    final_probed_context = [final_probed_context[i] for i in mem_idx]

    for i in range(len(final_memory_probes)):
        stacked_recog.append([recog_question,probe_ship,final_memory_probes[i]])

def pt2_memory_probes(choice_blocks=choice_blocks):
    """Running the seventh room and intermittent memory trials"""
    global pt2_index,study,bonus_money,bonus_correct
    for block in choice_blocks:
        for trial in range(block):
            for img_path in stacked_seven_room_pirates[pt2_index]: # Show all pirates and take responses
                stim = visual.ImageStim(win, image=img_path,size=(1.2,1.2))
                stim.draw()
            response_clock = core.Clock()
            island_clock = core.Clock()
            win.flip()
            resp_key = event.waitKeys(keyList=keyList,timeStamped=response_clock,maxWait=3)
            
            if resp_key:
                key,RT = resp_key[0] # RT used for data collection
                if keyList[0] in key: # 1
                    choice = 'red_pirate'
                    pirateChoice = stacked_seven_red[pt2_index]
                    pirateReward = stacked_seven_red_reward[pt2_index]
                if keyList[1] in key: # 2
                    choice = 'white_pirate'
                    pirateChoice = stacked_seven_white[pt2_index]
                    pirateReward = stacked_seven_white_reward[pt2_index]
                if keyList[2] in key: # 3
                    choice = 'black_pirate'
                    pirateChoice = stacked_seven_black[pt2_index]
                    pirateReward = stacked_seven_black_reward[pt2_index]
                if pt2_index < num_trials:
                    show_stacked_images(img_paths=pirateChoice,duration=1)
                    show_stacked_images(img_paths=pirateReward,duration=1)
                    show_stacked_images(img_paths=stacked_seven_room[pt2_index],duration=1)
                    pt2_index +=1
                    study.append({
                        "ID": "",
                        "TrialType":f"pirate_{pt2_index+1}",
                        "PayoutDistNum":"",
                        "BlockNum": "",
                        "contextOrder": "",
                        "reward_rate_red":"",
                        "reward_rate_white":"",
                        "reward_rate_black":"",
                        "probe_order": "",
                        "QuizFailedNum": "",
                        "TimeElapsed": experiment_clock.getTime(),
                        "key_press": key,
                        "RT": RT,
                        "context": "blank",
                        "reward_prob_red":probabilities[1][pt2_index],
                        "reward_prob_white":probabilities[2][pt2_index],
                        "reward_prob_black":probabilities[3][pt2_index],
                        "choice":choice,
                        "probe":"",
                        "reward":ifReward[int(key)][pt2_index],
                        "confidence":"",
                        "TimeInBlock": island_clock.getTime(),
                        "Bonus":""
                    })
                    write_study()  
                else: pass
            else:
                too_slow()
        get_memory_probe()
    bonus_money = int(np.round(bonus_correct*0.25))

def get_memory_probe():
    """Making the pt 2 probe memory questions"""
    global probed_mem_trial,final_memory_probes,old_probe_list,new_probe_list,study,bonus_correct
    for img_path in stacked_recog[probed_mem_trial]: # Show all pirates and take responses
        stim = visual.ImageStim(win, image=img_path,size=(1.2,1.2))
        stim.draw()
    response_clock = core.Clock()
    island_clock = core.Clock()
    win.flip()
    resp_key = event.waitKeys(keyList=recogKeyList,timeStamped=response_clock,maxWait=3)
    choice = 'none' # If no response
    if resp_key:
        key,RT = resp_key[0] # RT used for data collection
        if recogKeyList[0] in key: # 5
            choice = 'sure_old'
            if final_memory_probes[probed_mem_trial] in old_probe_list:
                correct = 1
                bonus_correct += 1
                confidence = 'sure'
                show_stacked_images(stacked_recog[probed_mem_trial] + [reward],duration=1)
            elif final_memory_probes[probed_mem_trial] in new_probe_list:
                correct = 0
                confidence = 'sure'
                show_stacked_images(stacked_recog[probed_mem_trial] + [no_reward],duration=1)
        if recogKeyList[1] in key: # 6
            choice = 'unsure_old'
            if final_memory_probes[probed_mem_trial] in old_probe_list:
                correct = 1
                bonus_correct += 1
                confidence = 'unsure'
                show_stacked_images(stacked_recog[probed_mem_trial] + [reward],duration=1)
            elif final_memory_probes[probed_mem_trial] in new_probe_list:
                correct = 0
                confidence = 'unsure'
                show_stacked_images(stacked_recog[probed_mem_trial] + [no_reward],duration=1)
        if recogKeyList[2] in key: # 7
            choice = 'unsure_new'
            if final_memory_probes[probed_mem_trial] in old_probe_list:
                correct = 0
                confidence = 'unsure'
                show_stacked_images(stacked_recog[probed_mem_trial] + [no_reward],duration=1)
            elif final_memory_probes[probed_mem_trial] in new_probe_list:
                correct = 1
                bonus_correct += 1
                confidence = 'unsure'
                show_stacked_images(stacked_recog[probed_mem_trial] + [reward],duration=1)
        if recogKeyList[3] in key: # 8
            choice = 'sure_new'
            if final_memory_probes[probed_mem_trial] in old_probe_list:
                correct = 0
                confidence = 'sure'
                show_stacked_images(stacked_recog[probed_mem_trial] + [no_reward],duration=1)
            elif final_memory_probes[probed_mem_trial] in new_probe_list:
                correct = 1
                bonus_correct += 1
                confidence = 'sure'
                show_stacked_images(stacked_recog[probed_mem_trial] + [reward],duration=1)
        probed_mem_trial +=1
        study.append({
            "ID": "",
            "TrialType":f"pirate_recog",
            "PayoutDistNum":"",
            "BlockNum": "",
            "contextOrder": "",
            "reward_rate_red":"",
            "reward_rate_white":"",
            "reward_rate_black":"",
            "probe_order": "",
            "QuizFailedNum": "",
            "TimeElapsed": experiment_clock.getTime(),
            "key_press": key,
            "RT": RT,
            "context": final_probed_context[probed_mem_trial],
            "reward_prob_red":"",
            "reward_prob_white":"",
            "reward_prob_black":"",
            "choice":choice,
            "probe":"",
            "reward":correct,
            "confidence":confidence,
            "TimeInBlock": island_clock.getTime(),
            "Bonus":""
        })
        write_study()  
    else:
        too_slow()
        probed_mem_trial +=1
            
stacked_source_memory = []
stacked_source_memory_reward = []
stacked_source_memory_no_reward = []
filtered_context = []
def source_memory_init():
    """Initializing source memory"""
    global old_probe_list,filtered_context
    for i,probe in enumerate(old_probe_list):
        stacked_source_memory.append([source_question,probe_ship,probe]+source_memory_contexts)
        stacked_source_memory_reward.append([source_question,probe_ship,probe,reward]+source_memory_contexts)
        stacked_source_memory_no_reward.append([source_question,probe_ship,probe,no_reward]+source_memory_contexts)
    filtered_context = [val for val in final_probed_context if val != "NA"]

source_memory_trial = 0

def pt2_source_memory():
    """Source memory trials in part 2"""
    global stacked_source_memory,stacked_source_memory_reward,stacked_source_memory_no_reward,source_memory_trial
    for img_path in stacked_source_memory[source_memory_trial]: # Show all pirates and take responses
        stim = visual.ImageStim(win, image=img_path,size=(1.2,1.2))
        stim.draw()
    response_clock = core.Clock()
    island_clock = core.Clock()
    win.flip()
    resp_key = event.waitKeys(keyList=source_key_list,timeStamped=response_clock,maxWait=3)
    choice = 'none' # If no response
    if resp_key:
        key,RT = resp_key[0] # RT used for data collection
        if source_key_list[0] in key: # 1
            choice = 'context_1'
            if filtered_context[source_memory_trial]=='1':
                correct = 1
                show_stacked_images(stacked_source_memory[source_memory_trial] + [reward],duration=1)
            else:
                correct = 0
                show_stacked_images(stacked_source_memory[source_memory_trial] + [no_reward],duration=1)
        if source_key_list[1] in key: # 2
            choice = 'context_2'
            if filtered_context[source_memory_trial]=='2':
                correct = 1
                show_stacked_images(stacked_source_memory[source_memory_trial] + [reward],duration=1)
            else:
                correct = 0
                show_stacked_images(stacked_source_memory[source_memory_trial] + [no_reward],duration=1)
        if source_key_list[2] in key: # 3
            choice = 'context_3'
            if filtered_context[source_memory_trial]=='3':
                correct = 1
                show_stacked_images(stacked_source_memory[source_memory_trial] + [reward],duration=1)
            else:
                correct = 0
                show_stacked_images(stacked_source_memory[source_memory_trial] + [no_reward],duration=1)
        if source_key_list[3] in key: # 4
            choice = 'context_4'
            if filtered_context[source_memory_trial]=='4':
                correct = 1
                show_stacked_images(stacked_source_memory[source_memory_trial] + [reward],duration=1)
            else:
                correct = 0
                show_stacked_images(stacked_source_memory[source_memory_trial] + [no_reward],duration=1)
        if source_key_list[4] in key: # 5
            choice = 'context_5'
            if filtered_context[source_memory_trial]=='5':
                correct = 1
                show_stacked_images(stacked_source_memory[source_memory_trial] + [reward],duration=1)
            else:
                correct = 0
                show_stacked_images(stacked_source_memory[source_memory_trial] + [no_reward],duration=1)
        if source_key_list[5] in key: # 6
            choice = 'context_6'
            if filtered_context[source_memory_trial]=='6':
                correct = 1
                show_stacked_images(stacked_source_memory[source_memory_trial] + [reward],duration=1)
            else:
                correct = 0
                show_stacked_images(stacked_source_memory[source_memory_trial] + [no_reward],duration=1)
        study.append({
            "ID": "",
            "TrialType":f"source_memory",
            "PayoutDistNum":"",
            "BlockNum": "",
            "contextOrder": "",
            "reward_rate_red":"",
            "reward_rate_white":"",
            "reward_rate_black":"",
            "probe_order": "",
            "QuizFailedNum": "",
            "TimeElapsed": experiment_clock.getTime(),
            "key_press": key,
            "RT": RT,
            "context": stacked_source_memory[source_memory_trial],
            "reward_prob_red":"",
            "reward_prob_white":"",
            "reward_prob_black":"",
            "choice":choice,
            "probe":"",
            "reward":correct,
            "confidence":"",
            "TimeInBlock": island_clock.getTime(),
            "Bonus":""
        })
        source_memory_trial +=1
        if source_memory_trial == len(stacked_source_memory):
            pass
        else: 
            show_blank_screen()
            pt2_source_memory()
    else:
        too_slow()
        source_memory_trial +=1
        if source_memory_trial == len(stacked_source_memory):
            pass
        else: 
            show_blank_screen()
            pt2_source_memory()

def show_blank_screen(duration=0.5):
    """How long a blank white screen shows"""
    win.flip()  # show the blank screen
    core.wait(duration)  # duration in seconds

def pt2_best_pirate():
    global best_pirate_trial
    """For last phase where each context is shown with a pirate selection"""
    y = [0.1,0.1,-0.2]
    for i,img_path in enumerate(stacked_best_pirate[best_pirate_trial]): # Show all pirates and take responses
        stim = visual.ImageStim(win, image=img_path,size=(1.2,1.2),pos = (0,y[i]))
        stim.draw()
    response_clock = core.Clock()
    island_clock = core.Clock()
    win.flip()
    resp_key = event.waitKeys(keyList=keyList,timeStamped=response_clock,maxWait=2)
    choice = "none" # No response
    if resp_key:
        key,RT = resp_key[0] # RT used for data collection
        if source_key_list[0] in key: # 1
            choice = 'red_pirate'
            show_stacked_images(stacked_best_pirate[best_pirate_trial] + [red_best],duration=1, y=[0.1,0.1,-0.2,-0.2])
        if source_key_list[1] in key: # 2
            choice = 'white_pirate'
            show_stacked_images(stacked_best_pirate[best_pirate_trial] + [white_best],duration=1, y = [0.1,0.1,-0.2,-0.2])
        if source_key_list[2] in key: # 3
            choice = 'black_pirate'
            show_stacked_images(stacked_best_pirate[best_pirate_trial] + [black_best],duration=1, y = [0.1,0.1,-0.2,-0.2])
        study.append({
            "ID": "",
            "TrialType":f"source_memory",
            "PayoutDistNum":"",
            "BlockNum": "",
            "contextOrder": "",
            "reward_rate_red":"",
            "reward_rate_white":"",
            "reward_rate_black":"",
            "probe_order": "",
            "QuizFailedNum": "",
            "TimeElapsed": experiment_clock.getTime(),
            "key_press": key,
            "RT": RT,
            "context": context_order_labels[best_pirate_trial],
            "reward_prob_red":"",
            "reward_prob_white":"",
            "reward_prob_black":"",
            "choice":choice,
            "probe":"",
            "reward":"",
            "confidence":"",
            "TimeInBlock": island_clock.getTime(),
            "Bonus":""
        })
        write_study()  
        best_pirate_trial +=1
        if best_pirate_trial == len(contexts):
            pass
        else: 
            show_blank_screen()
            pt2_best_pirate()
    else:
        too_slow()
        best_pirate_trial +=1
        if best_pirate_trial == len(contexts):
            pass
        else: 
            show_blank_screen()
            pt2_best_pirate()

# How data is saved to CSV
def save_data(participant_id, trials,end=False):
    """Save collected data to a CSV file, automatically detecting headers."""
    global study,bonus_money
    if end:
        study.append({
                "ID": "",
                "TrialType":"EndStudy",
                "PayoutDistNum":"",
                "BlockNum": "",
                "contextOrder": "",
                "reward_rate_red":"",
                "reward_rate_white":"",
                "reward_rate_black":"",
                "probe_order": "",
                "QuizFailedNum": "",
                "TimeElapsed": experiment_clock.getTime(),
                "key_press": "",
                "RT": "",
                "context": "",
                "reward_prob_red":"",
                "reward_prob_white":"",
                "reward_prob_black":"",
                "choice":"",
                "probe":"",
                "reward":"",
                "confidence":"",
                "TimeInBlock": "",
                "Bonus":bonus_money
            })
    write_study()  
    folder_name = "data"
    os.makedirs(folder_name, exist_ok=True)  # Uses data directory and checks if it exists before adding

    filename = os.path.join(folder_name, f"participant_{participant_id}.csv") # Data is saved under the id

    if not trials: # checks if it is empty
        print("No trial data to save.")
        return 

    headers = trials[0].keys()  # Header is first study keys (should be consistent accross study calls)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for trial in trials: # Every row is a trial
            writer.writerow(trial) 

    print(f"Data saved to {filename}") # Confirmation that data saved



# Main experiment flow

show_text(welcome_txt)

show_text(different_places,image_path= all_contexts,height=0.3)

show_text(goal_of_game_1,image_path=tutorial_ship,height=0.3)

show_multi_img_text([goal_of_game_2a,goal_of_game_2b,goal_of_game_2c,goal_of_game_2d],image_paths=[tutorial_all_pirates,tutorial_reward,tutorial_noreward],heights=[0.8,0.25,-0.25,-0.8],img_pos=[0.5,0,-0.5],x=0.3,y=0.4)

show_text(probabilistic,image_path=tutorial_blue_pirate,height=0.5,keys=['1'])

practice_blue_loop()

show_text(blue_beard_outcome,keys=['space'])

practice_pirates()

practice_pirates(text=pick_pirate_again,switch='nowin')

show_text(text=time_out,height=0.5,image_path=timeout_img)

show_text(text=probe,height=0.6,image_path=example_probe,text_height=0.05)

show_text(text=changepoint)

show_text(text=drift,height=0.4,image_path=contingency,img_pos=-0.4)

show_text(text=summary)

show_stacked_images(desert_welcome,duration=3)

practice_pirate_loop()

show_text(quiz_intro)

run_quiz()

show_text("Good job! You’re now ready to move on to the real game! Remember this game will be difficult but don't get discouraged and try your best!" + space_bar)

save_data(participant_id,study)

learn_phase_loop()

save_data(participant_id,study)

show_text("You are all done with the first part of the study! Thank you for participating.\n\n\n\nPress the spacebar to continue")
take_break()

show_multi_img_text([goal_summary,goal_summary2,space_bar],image_paths=[tutorial_reward,tutorial_noreward],heights=[0.6,-0.1,-0.7],img_pos=[0.2,-0.4],x=0.3,y=0.4)

show_text(how_to_pick_summary)

show_text(text=recognition_1,image_path=example_recog,height=0.5)

show_text(text=recognition_2,image_path=example_recog,height=0.4,img_pos=-0.5,text_height=0.06)

show_multi_img_text(texts=[recognition_3,recognition_3b,recognition_3c],image_paths=[memory_reward,tutorial_noreward],heights=[0.6,0.1,-0.6],img_pos=[0.35,-0.20],x=0.3,y=0.4)

show_text(begin_final)
init_responses()
pt2_memory_probes(choice_blocks=choice_blocks)

show_text(text=source_memory + space_bar,height=0.5,image_path=pt2_source_memory_img,img_pos=-0.4)

source_memory_init()
pt2_source_memory()

show_text(pick_best_pirate + space_bar,height=0.45, image_path=pick_pirate_tutorial,img_pos=-0.4,x=1.4,y=0.8)

pt2_best_pirate()

save_data(participant_id,study,end=True)

show_text(f"Thank you for your participation in this expeirment. You have collected ${bonus_money} in bonus payment. Please contact your experimenter to let them know that you are all done and do not exit out of this page.")

win.close()
core.quit()

