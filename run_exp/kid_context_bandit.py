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

# Debug and block length
debug = config.get("debug", False)
block_length = config["block_length_debug"] if debug else config["block_length"]

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
desert_img = image_prefix + "contexts/context_desert.png"
cavern_welcome_text = image_prefix + "travel/welcome_cavern.png"
cavern_img = image_prefix + "contexts/context_cavern.png"
remember_text_img = image_prefix + "miscellaneous/remember.png"
all_pirates = image_prefix + "pirates/pirates_all.png"
red_pirate = image_prefix + "pirates/red_beard.png"
white_pirate = image_prefix + "pirates/white_beard.png"
black_pirate = image_prefix + "pirates/black_beard.png"

reward = image_prefix + "rewards/reward.png"
no_reward = image_prefix + "rewards/reward_no.png"

probe_ship = image_prefix + "miscellaneous/cargo_ship.png"
bye_island = image_prefix + "travel/bye.png"

best_pirate = image_prefix + "tutorial/prac_best_pirate.png"
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

# Instruction init
space_bar = "\n\n[Press the space bar to continue]"
welcome_txt = "Welcome! This is the first part of the study. It will last ~30 minutes. In part 1 and part 2 of the study, you are the head captain of a pirate ship traveling around the world to different islands. You will play a few different games throughout these two parts. Here is an overview of them:\n\nPart 1 (today)\n\n1. Instructions, practice game, and quiz\n\n2. Pick a pirate game on 6 different islands\n\nPart 2 (tomorrow)\n\n3. Pick a pirate game and memory game\n\n4. Where did you see this ship? " + space_bar
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

# Reward
import numpy as np

# ----- Constants -----
num_bandits = 3
first_block = 30
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

probabilities = {
    1: payout[0] * 0.01,
    2: payout[1] * 0.01,
    3: payout[2] * 0.01
}

ifReward = {key: np.random.binomial(1, prob) for key, prob in probabilities.items()}

reward_imgs = {
    key: np.where(value == 1, reward, no_reward)
    for key, value in ifReward.items()
}

# Valid probes (seen in task) and Invalid (foils during memory phase)
valid_probe_images = []
invalid_probe_images = []
for i in range(1,231): # 230 trials (30 trials first block + 40 trials * 5 other blocks)
    if i < 10:   
        valid_probe_images.append(image_prefix + f"probes/probes-0{i}.png")
    else:
        valid_probe_images.append(image_prefix + f"probes/probes-{i}.png")
for i in range(231,256): # 25 trials (wrong probe trials during testing)
    invalid_probe_images.append(image_prefix + f"probes/probes-{i}.png")

random.shuffle(valid_probe_images)
random.shuffle(invalid_probe_images)

# Context-based trial order

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

for context_idx,context in enumerate(contexts):
    if context_idx == 0:
        for trial_idx in range(first_block):
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
    else:
        for trial_idx in range(block_len):
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

# Sequences
ship_sequence = [image_prefix + r for r in config["ship_sequence"]]

# Randomized planets
planet_prefix = np.random.choice(np.arange(1,121), 120, replace=False)
planets = [image_prefix + f"aliens/alien_planet-{planet}.jpg" for planet in planet_prefix]
gem = 0
decay = None
alien_index = 0
prt_clock = core.Clock() # Initialized prt timer
block_time = 0
text_index = 0
failedNum = 0
index = 0
first_planet = True

keyList = [config['params']['BUTTON_1'],config['params']['BUTTON_2'],config['params']['BUTTON_3']]

if len(sys.argv) < 2: # Call participant ID in terminal
    print("Error: No participant ID provided.")
    print("Usage: python3 experiment.py <participant_id>")
    sys.exit(1)  # Exit with error code 1

# Get participant ID from command-line argument
participant_id = sys.argv[1]

study = []

# For study.yaml output
def study_filename(subject_id_str):
    extension = ".yaml"
    if not subject_id_str:
        return 'study' + extension
    return 'study' + "." + str(subject_id_str) + extension

def write_study():
    if not participant_id:
        raise Exception("Shouldn't write to original study file! Please provide a valid subject ID.")
    with open(study_filename(participant_id), 'w') as file:
        for row in study:
            if isinstance(row.get("AlienOrder"), np.ndarray):
                row["AlienOrder"] = row["AlienOrder"].tolist()
        yaml.dump(study, file)

# Initial data format -- What the csv reader takes as column names
study.append({
    "ID": participant_id,
    "TrialType":f"InitializeStudy",
    "BlockNum": "",
    "AlienOrder": planet_prefix,
    "QuizResp": "",
    "QuizFailedNum": "",
    "TimeElapsed": experiment_clock.getTime(),
    "RT": "",
    "PRT": "",
    "Galaxy": "",
    "DecayRate": "",
    "AlienIndex": "",
    "GemValue": "",
    "TimeInBlock": ""
})
# init in case exp breaks
write_study()
# For practice quiz
questions = [
    "What affects the amount of bonus money you will earn?",
    "The length of this experiment",
    "You get to stay at home base as long as you like."
]

choices = [
    ["The number of planets you visit", "How long you stay at home base", "The number of gems you collect"],
    ["Depends on how many planets you've visited", "Is fixed", "Depends on how many gems you've collected"],
    ["True", "False"]
]

correct_answers = ["The number of gems you collect", "Is fixed", "False"]

# For decay rate
def rbeta(alpha, beta):
    """Generates a random number from a Beta distribution."""
    return np.random.beta(alpha, beta)

# Takes galaxy type to know which distribution to create (poor, neutral, rich)
def get_decay_rate(galaxy):
    """Returns a decay rate sampled from a Beta distribution based on the galaxy type."""
    galaxy_distributions = {
        0: (13, 51),  # Rich planet
        1: (50, 50),  # Neutral planet
        2: (50, 12)   # Poor planet
    }
    
    if galaxy in galaxy_distributions:
        alpha, beta = galaxy_distributions[galaxy]
        return rbeta(alpha, beta)
    else:
        raise ValueError("Galaxy index out of range. Must be 0, 1, or 2.")


galaxy = None # Galaxy is initialized later and can randomly start between planet types

# If first is true it randomly chooses a planet type but if not it uses the 80% stay 20% switch
def get_galaxy(first=False):
    global galaxy
    if first is True:
        galaxy = np.random.randint(0,3)
    else:
        percentNum = np.random.randint(1,11)
        if percentNum > 8:
            newGalaxy = galaxy
            while newGalaxy == galaxy:
                newGalaxy = np.random.randint(0,3)
            galaxy = newGalaxy
        

def show_text(text, duration=0, image_path=None,x=1,y=1,height=0,img_pos = -0.3,text_height=config['params']['FONT_SIZE'],home=False,keys=['space']):
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
    event.waitKeys(keyList=keys)
    if home == False:
        study.append({
            "ID": "",
            "TrialType":f"Instruction_{text_index}",
            "BlockNum": "",
            "AlienOrder": "",
            "QuizResp": "",
            "QuizFailedNum": "",
            "TimeElapsed": experiment_clock.getTime(),
            "RT": "",
            "PRT": "",
            "Galaxy": "",
            "DecayRate": "",
            "AlienIndex": "",
            "GemValue": "",
            "TimeInBlock": ""
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
        "BlockNum": "",
        "AlienOrder": "",
        "QuizResp": "",
        "QuizFailedNum": "",
        "TimeElapsed": experiment_clock.getTime(),
        "RT": "",
        "PRT": "",
        "Galaxy": "",
        "DecayRate": "",
        "AlienIndex": "",
        "GemValue": "",
        "TimeInBlock": ""
                })
# Shows an image for some duration
def show_image(img_path, duration=1.5):
    """Displays an image for a given duration"""
    stim = visual.ImageStim(win, image=img_path,size=(1.2,1.2))
    stim.draw()
    win.flip()
    core.wait(duration)

def show_stacked_images(img_paths = [], duration=3):
    """Displays an image for a given duration"""
    for img_path in img_paths:
        stim = visual.ImageStim(win, image=img_path,size=(1.2,1.2))
        stim.draw()
    win.flip()
    core.wait(duration)

# Timeout image for 2 seconds
def too_slow():
    show_image(timeout_img,duration=2)

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
        "TrialType":f"Travel_Planet",
        "BlockNum": "",
        "AlienOrder": "",
        "QuizResp": "",
        "QuizFailedNum": "",
        "TimeElapsed": experiment_clock.getTime(),
        "RT": "",
        "PRT": "",
        "Galaxy": "",
        "DecayRate": "",
        "AlienIndex": "",
        "GemValue": "",
        "TimeInBlock": ""
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
        pass
    else:
        show_text("Oops, you missed some questions. Now that you’ve heard the correct answers. Try the quiz again!" + space_bar)
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

def practice_pirate_loop(duration = 2,setting = 'desert'):
    """For choosing the pirate, getting the probe, and seeing if there is a reward"""
    global curr_prac_trial
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
            set_img = cavern_img
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
                if keyList[1] in bestkeys: # 2
                    show_text("That's incorrect! Red beard was the best."+ space_bar)
                if keyList[2] in bestkeys: # 3
                    show_text("That's incorrect! Red beard was the best."+ space_bar)

            stim = visual.ImageStim(win, image=source_practice,size=(1.2,1.2))
            stim.draw()
            win.flip()
            source_key = event.waitKeys(keyList=[keyList[0],keyList[1]])

            if best_key:
                sourcekey = source_key[0]
                if keyList[0] in sourcekey: # 1
                    show_text("That's incorrect! You saw this ship on the cavern island."+ space_bar)
                if keyList[1] in sourcekey: # 2
                    show_text("That's correct! You saw this ship on the cavern island." + space_bar)

    else:
        show_image(timeout_img,duration=2)
        practice_pirate_loop()

def learn_phase_loop():
    """For choosing the pirate, getting the probe, and seeing if there is a reward"""
    global curr_trial
    planet_shift_indx = [0,30,70,110,150,190] # 0 index where contexts shift
    if curr_trial in planet_shift_indx:
        show_stacked_images(stacked_planet_welcome[curr_trial],duration=3) # Show welcome on first visit
    for img_path in stacked_all_pirates[curr_trial]: # Show all pirates and take responses
        stim = visual.ImageStim(win, image=img_path,size=(1.2,1.2))
        stim.draw()
    response_clock = core.Clock()
    win.flip()
    resp_key = event.waitKeys(keyList=keyList,timeStamped=response_clock,maxWait=2)
    
    if resp_key:
        key,RT = resp_key[0] # RT used for data collection
        if keyList[0] in key: # 1
            pirateChoice = stacked_red_pirate[curr_trial]
            pirateProbe = stacked_red_remember[curr_trial]
            pirateReward = stacked_red_reward[curr_trial]
        if keyList[1] in key: # 2
            pirateChoice = stacked_white_pirate[curr_trial]
            pirateProbe = stacked_white_remember[curr_trial]
            pirateReward = stacked_white_reward[curr_trial]
        if keyList[2] in key: # 3
            pirateChoice = stacked_black_pirate[curr_trial]
            pirateProbe = stacked_black_remember[curr_trial]
            pirateReward = stacked_black_reward[curr_trial]
        show_stacked_images(img_paths=pirateChoice,duration=1)
        show_stacked_images(img_paths=pirateProbe,duration=2)
        show_stacked_images(img_paths=pirateReward,duration=1)
        show_stacked_images(img_paths=stacked_island_nopirate[curr_trial],duration=1)
        curr_trial +=1 # Advance trial
        if curr_trial == first_block + (block_len * (num_blocks - 1)):
            show_stacked_images(stacked_island_bye[curr_trial-1],duration=1.5)
        elif curr_trial in planet_shift_indx:
                show_stacked_images(stacked_island_bye[curr_trial-1],duration=1.5)
                take_break()
                travel_trial()
                learn_phase_loop()
        else:
            learn_phase_loop()
    else:
        show_image(timeout_img,duration=2)
        curr_trial -= 1
        learn_phase_loop()

def take_break():
    """Breaks between islands"""
    show_text(text="Time to take a quick break! You have 2 minutes to rest, but you can move on sooner if you'd like."+space_bar,duration=120)



# How data is saved to CSV
def save_data(participant_id, trials):
    """Save collected data to a CSV file, automatically detecting headers."""
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

# show_text(different_places,image_path= all_contexts,height=0.3)

# show_text(goal_of_game_1,image_path=tutorial_ship,height=0.3)

# show_multi_img_text([goal_of_game_2a,goal_of_game_2b,goal_of_game_2c,goal_of_game_2d],image_paths=[tutorial_all_pirates,tutorial_reward,tutorial_noreward],heights=[0.8,0.25,-0.25,-0.8],img_pos=[0.5,0,-0.5],x=0.3,y=0.3)

# show_text(probabilistic,image_path=tutorial_blue_pirate,height=0.5,keys=['1'])

# practice_blue_loop()

# show_text(blue_beard_outcome,keys=['space'])

# practice_pirates()

# practice_pirates(text=pick_pirate_again,switch='nowin')

# show_text(text=time_out,height=0.5,image_path=timeout_img)

# show_text(text=probe,height=0.6,image_path=example_probe,text_height=0.05)

# show_text(text=changepoint)

show_text(text=drift,height=0.4,image_path=contingency,img_pos=-0.4)

# show_text(text=summary)

# show_stacked_images(desert_welcome,duration=3)

# practice_pirate_loop()

# show_text(quiz_intro)

# run_quiz()

# show_text("Good job! You’re now ready to move on to the real game! Remember this game will be difficult but don't get discouraged and try your best!" + space_bar)

learn_phase_loop()

# save_data(participant_id,study)
show_text("You are all done with the first part of the study! Thank you for participating.\n\n\n\nPress the spacebar to exit")

win.close()
core.quit()
