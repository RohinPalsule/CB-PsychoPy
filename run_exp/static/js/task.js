'use strict';

////////// PAVLOVIA ////////////
var pavlovia_init = {
	type: "pavlovia",
	command: "init"
};

var pavlovia_finish = {
	type: "pavlovia",
	command: "finish"
	};
///////////////////////////////


// Initialize canvas
var canvas = $('#canvasdiv');

// get subj id from url
var urlvar = jsPsych.data.urlVariables()
var participant_id = jsPsych.data.getURLVariable('participant_ID');
var subject_id = jsPsych.data.getURLVariable('subject_ID');
var age = jsPsych.data.getURLVariable('age');
var gender = jsPsych.data.getURLVariable('gender');


function makeObject(img_path,id) {
  // create the image
	let image = $('<img>', {
	    src: img_path,
			width:"1000px",
	      // width: SIZE,
	   })
	  .appendTo(canvas)
		.css("position","absolute") // always string
		.css("padding-top","100px") // always string
		.css("left","0") // always string
		.css("right","0") // always string
		.css("margin-left","auto") // always string
		.css("margin-right","auto") // always string
  // add to canvas once loaded
  //image.onload = function(){
    //ctx.drawImage(image,canvas.width / 2 - (image.width*0.6) / 2,
    //canvas.height - (image.height*0.6)/1.0,image.width*0.6, image.height*0.6);
//  }

  window.image = image;

  return {
    hide() {
      image.hide();
    },
    show () {
      image.show();
    },
    image: image,
    id: id,
  }
};

function randomize(array) {
  array = array
    .map((value) => ({ value, sort: Math.random() }))
    .sort((a, b) => a.sort - b.sort)
    .map(({ value }) => value)
    return array;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
var deck = makeObject('run_exp/static/images/miscellaneous/deck.png','deck') // stays constant thru out exp

let contextArray_true = [makeObject('run_exp/static/images/contexts/context_coast.png','coast'),
makeObject('run_exp/static/images/contexts/context_countryside.png','countryside'),
makeObject('run_exp/static/images/contexts/context_mountain.png','mountain'),
makeObject('run_exp/static/images/contexts/context_forest.png','forest'),
makeObject('run_exp/static/images/contexts/context_highway.png','highway'),
makeObject('run_exp/static/images/contexts/context_city.png','city')]
let contextArray_ind = [0,1,2,3,4,5]
contextArray_ind = randomize(contextArray_ind);
let contextArray = []
for (let i = 0; i < 6; i++) {
  contextArray.push(contextArray_true[contextArray_ind[i]])
}

contextArray.push(makeObject('run_exp/static/images/contexts/context_blank.png','blank'))
contextArray.push(makeObject('run_exp/static/images/contexts/context_desert.png','desert'))
contextArray.push(makeObject('run_exp/static/images/contexts/context_cavern.png','cavern'))

contextArray.forEach(context => context.hide())


var cargo_ship = makeObject('run_exp/static/images/miscellaneous/cargo_ship.png','cargo_ship');

let shipArray = [makeObject('run_exp/static/images/travel/ship_up.png','ship_up'), // ship up
makeObject('run_exp/static/images/travel/ship_down.png','ship_down')] // ship down] // map for welcome
shipArray.forEach(item => item.hide())

let pirateArray = [makeObject('run_exp/static/images/pirates/pirates_all.png','all'), // all pirates
makeObject('run_exp/static/images/pirates/red_beard.png','red'), // red beard
makeObject('run_exp/static/images/pirates/white_beard.png','white'), // white beard
makeObject('run_exp/static/images/pirates/black_beard.png','black')] // black beard
pirateArray.forEach(pirate => pirate.hide())

let rewardArray = [makeObject('run_exp/static/images/rewards/reward_no.png','noReward'), // nothing
makeObject('run_exp/static/images/rewards/reward.png','reward'),
makeObject('run_exp/static/images/rewards/reward_small.png','small_reward'),
makeObject('run_exp/static/images/rewards/money_bag.png','money_bag')] // money bag
rewardArray.forEach(reward => reward.hide())


// make array of probe images
let valid_probe_images_true = [];
let invalid_probe_images_true = [];
let prac_probe_images = [];

for (let i = 1; i < 10; i++) {
  valid_probe_images_true.push(makeObject('run_exp/static/images/probes/probes-0'+i+'.png','probe-0'+i))
}

for (let i = 10; i < 231; i++) {
  valid_probe_images_true.push(makeObject('run_exp/static/images/probes/probes-'+i+'.png','probe-'+i))
}


for (let i = 231; i < 256; i++) {
  invalid_probe_images_true.push(makeObject('run_exp/static/images/probes/probes-'+i+'.png','probe-'+i))
}

for (let i = 256; i < 266; i++) {
  prac_probe_images.push(makeObject('run_exp/static/images/probes/probes-'+i+'.png','probe-'+i))
}


// randomize the array of images
let valid_probe_images_ind = randomize([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229]); // decision_ttrial --> proobe
let invalid_probe_images_ind = randomize([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16,17,18,19,20,21,22,23,24]);

let valid_probe_images = [];
let invalid_probe_images = [];


for (let i = 0; i < 230; i++) {
  valid_probe_images.push(valid_probe_images_true[valid_probe_images_ind[i]])
}

for (let i = 0; i < 25; i++) {
  invalid_probe_images.push(invalid_probe_images_true[invalid_probe_images_ind[i]])
}

// hide the array of images
console.log(valid_probe_images)
valid_probe_images.forEach(image => image.hide())
invalid_probe_images.forEach(image => image.hide())
prac_probe_images.forEach(image => image.hide())


// intialize array for saving context image # on each learn phase (contexts 1-6)
var contexts = [];

var current_trial = 0;
var current_context = -1;
var context_pick_n = 0;
var current_prac_trial = 0;
var corr_old_new_judge = 0;


let miscellArray = [makeObject('run_exp/static/images/probes/gas.png','probe'), // probe
makeObject('run_exp/static/images/miscellaneous/remember.png','remember'), // remember
makeObject('run_exp/static/images/miscellaneous/hurry_up.png','hurryUp'), // hurry up
makeObject('run_exp/static/images/miscellaneous/probe_recog.png','confidence_rate'),//confidence ratings
deck,
contextArray[7],
makeObject('run_exp/static/images/rewards/feedback_correct.png','correct'),
makeObject('run_exp/static/images/rewards/feedback_incorrect.png','incorrect'),
makeObject('run_exp/static/images/travel/ahoy.png','ahoy'),
makeObject('run_exp/static/images/travel/bye.png','bye'),cargo_ship,
makeObject('run_exp/static/images/miscellaneous/best_pirate.png','best_pirate')]
miscellArray.forEach(item => item.hide())

let welcomeArray_true = [makeObject('run_exp/static/images/travel/welcome_coast.png','welcome_coast'),
makeObject('run_exp/static/images/travel/welcome_meadow.png','welcome_meadow'),
makeObject('run_exp/static/images/travel/welcome_mountain.png','welcome_mountain'),
makeObject('run_exp/static/images/travel/welcome_forest.png','welcome_forest'),
makeObject('run_exp/static/images/travel/welcome_road.png','welcome_road'),
makeObject('run_exp/static/images/travel/welcome_city.png','welcome_city')]
let welcomeArray = []
for (let i = 0; i < 6; i++) {
  welcomeArray.push(welcomeArray_true[contextArray_ind[i]])
}
welcomeArray.push(makeObject('run_exp/static/images/travel/welcome_final.png','welcome_final'))
welcomeArray.push(makeObject('run_exp/static/images/travel/welcome_desert.png','welcome_desert'))
welcomeArray.push(makeObject('run_exp/static/images/travel/welcome_cavern.png','welcome_cavern'))

welcomeArray.forEach(item => item.hide())

let islandTrackerArray = [
makeObject('run_exp/static/images/contexts/island_1.png','island_1'),
makeObject('run_exp/static/images/contexts/island_2.png','island_2'),
makeObject('run_exp/static/images/contexts/island_3.png','island_3'),
makeObject('run_exp/static/images/contexts/island_4.png','island_4'),
makeObject('run_exp/static/images/contexts/island_5.png','island_5'),
makeObject('run_exp/static/images/contexts/island_6.png','island_6')];

islandTrackerArray.forEach(item => item.hide())



var probabilities = { 1: payout[0].map(x=>x*0.01),
  2: payout[1].map(x=>x*0.01),
  3: payout[2].map(x=>x*0.01)
}

var pRew_Red = probabilities[1];
var pRew_White = probabilities[2];
var pRew_Black = probabilities[3];

var prac_prob = { 1: [1,1,1,0,1,1,1,1,0,1],
  2: [0,1,0,1,0,0,0,1,0,0,],
  3: [0,0,1,0,0,0,1,0,1,1]
}


var response_key_dict = {
  49: '1',
  50: '2',
  51: '3',
  52: '4',
  53: '5',
  54: '6',
  55: '7',
  56: '8'
}

var welcome_prac = {
  type: 'do_welcome',
  stimuli: miscellArray,
  trial_duration:4000,
	on_start: function (welcome) {
		if (current_prac_trial == 0) {
			miscellArray[5] = contextArray[7];
			welcome.text = welcomeArray[7];
		} else {
			miscellArray[5] = contextArray[8];
			welcome.text = welcomeArray[8];
		}
		welcome.stimuli = miscellArray;
	}
}

var intertrial = {
	type: 'html-keyboard-response',
  stimulus:"<p></p>",
  choices: jsPsych.NO_KEYS,
	trial_duration: 1000
}

var welcome = {
  type: 'do_welcome',
  stimuli: miscellArray,
  trial_duration:4000,
	on_start: function (welcome) {
		current_context += 1; // next this in welcome
		miscellArray[5] = contextArray[current_context];
		welcome.stimuli = miscellArray;
		welcome.text = welcomeArray[current_context];
		welcome.island_num = islandTrackerArray[current_context];
	}
}

var take_break = {
	type: 'html-keyboard-response',
  stimulus:"<div class='center'><p>Time to take a quick break! You have 2 minutes to rest, but you can move on sooner if you'd like.</p><p> Press the <b>space bar</b> to continue.</p></div>",
	trial_duration:120000,
  choices: ['space'],
	on_finish: function(data) {
		data.trial_type="break"
	}
}

var bye = {
  type: 'do_bye',
  stimuli: miscellArray,
  trial_duration:1500
}

var travel = {
  type: 'do_travel',
  stimuli: shipArray,
  choices:jsPsych.NO_KEYS,
  frame_time: 1500
}

var pick_best = {
  type: 'do_pick_best_pirate',
  choices: ['1','2','3'],
  pirates: pirateArray,
  context: null,
  miscell: miscellArray,
  on_start: function(pick_best) {
    let context_im = contextArray[context_pick_n];
    pick_best.context = context_im;
    pick_best.context_n = context_pick_n;
    pick_best.context_id = context_im.id;
},
on_finish: function(data) {
  let last_trial_data = jsPsych.data.get().last(1).values()[0];
  data.context_n = last_trial_data.context_n;
  data.context_id = last_trial_data.context_id;
  context_pick_n += 1;
}};

var prac_choice_trial = {
  type: 'do_trial',
  choices: ['1','2','3'],
  prompt:'<p>which pirate do you want to rob the next ship?</p>',
  trial_duration: 3000,
  pirates: pirateArray,
  rewards: rewardArray,
  miscell: null,
  probes: prac_probe_images,
  on_start: function(choice_trial) {
    miscellArray[0] = prac_probe_images[current_prac_trial];

		let potential_outcomes = [];
    for (let i = 1; i < 4; i++) {
      var outcome = Sampling.Bernoulli(prac_prob[i][current_prac_trial]).draw()
      potential_outcomes.push(outcome)
    }

    choice_trial.choice_outcomes = potential_outcomes;
    choice_trial.miscell = miscellArray;
    choice_trial.data = {choice_outcomes : potential_outcomes};
  },
  on_finish: function (data) {
    let last_trial_data = jsPsych.data.get().last(1).values()[0];
    let key_pressed =response_key_dict[data.response];

		if (key_pressed != null ) {
			current_prac_trial += 1;
		}
		if (current_prac_trial == 5) {
			jsPsych.addNodeToEndOfTimeline({timeline: [
				bye,travel,welcome_prac,prac_choice_trial],}, jsPsych.resumeExperiment);
		} else if (current_prac_trial == 9) {
			jsPsych.addNodeToEndOfTimeline({timeline: [intertrial,prac_choice_trial_2],}, jsPsych.resumeExperiment);
		} else if (current_prac_trial > 9) {
			jsPsych.addNodeToEndOfTimeline({timeline: [prac_best_pirate],}, jsPsych.resumeExperiment);
		} else {
			jsPsych.addNodeToEndOfTimeline({timeline: [intertrial,prac_choice_trial],}, jsPsych.resumeExperiment);
		}
  }
}

var prac_choice_trial_2 = {
  type: 'do_trial',
  choices: ['1','2','3'],
  prompt:'<p>which pirate do you want to rob the next ship?</p>',
  trial_duration: 3000,
  pirates: pirateArray,
  rewards: rewardArray,
  miscell: null,
  probes: prac_probe_images,
	after_blank: true,
  on_start: function(choice_trial) {
    miscellArray[0] = prac_probe_images[current_prac_trial];

		let potential_outcomes = [];
    for (let i = 1; i < 4; i++) {
      var outcome = Sampling.Bernoulli(prac_prob[i][current_prac_trial]).draw()
      potential_outcomes.push(outcome)
    }

    choice_trial.choice_outcomes = potential_outcomes;
    choice_trial.miscell = miscellArray;
    choice_trial.data = {choice_outcomes : potential_outcomes};
  },
  on_finish: function (data) {
    let last_trial_data = jsPsych.data.get().last(1).values()[0];
    let key_pressed =response_key_dict[data.response];

		if (key_pressed != null ) {
			current_prac_trial += 1;
		}
		if (current_prac_trial == 5) {
			jsPsych.addNodeToEndOfTimeline({timeline: [
				bye,travel,welcome_prac,prac_choice_trial],}, jsPsych.resumeExperiment);
		} else if (current_prac_trial == 9) {
			jsPsych.addNodeToEndOfTimeline({timeline: [intertrial,prac_choice_trial_2],}, jsPsych.resumeExperiment);
		} else if (current_prac_trial > 9) {
			jsPsych.addNodeToEndOfTimeline({timeline: [prac_best_pirate],}, jsPsych.resumeExperiment);
		} else {
			jsPsych.addNodeToEndOfTimeline({timeline: [intertrial,prac_choice_trial],}, jsPsych.resumeExperiment);
		}
  }
}

var choice_trial = {
  type: 'do_trial',
  choices: ['1','2','3'],
  prompt:'<p>which pirate do you want to rob the next ship?</p>',
  trial_duration: 3000,
  pirates: pirateArray,
  rewards: rewardArray,
  miscell: null,
  probes: valid_probe_images,
  on_start: function(choice_trial) {
    miscellArray[0] = valid_probe_images[current_trial];

    let potential_outcomes = [];
    for (let i = 1; i < 4; i++) {
      var outcome = Sampling.Bernoulli(probabilities[i][0]).draw()
      potential_outcomes.push(outcome)
    }

    choice_trial.choice_outcomes = potential_outcomes;
    choice_trial.miscell = miscellArray;
    choice_trial.data = {choice_outcomes : potential_outcomes};
  },
  on_finish: function (data) {
    // if no response was made need to remove this trial from available for probeing
    if (data.response == null) {
      availableForMemProbe = availableForMemProbe.filter((ar)=> ar != current_trial)
    }

    let last_trial_data = jsPsych.data.get().last(1).values()[0];
    let choice_outcomes = last_trial_data.choice_outcomes;
    let key_pressed =response_key_dict[data.response];

    data.context_n = current_context;
    contexts[current_trial] = current_context;
    data.context_img_id = contextArray[current_context].id;
    data.trial_n = current_trial;
		if (contextArray[current_context].id == 'blank') {
			data.probe_img_id = 'null';
		} else {
			data.probe_img_id = valid_probe_images[current_trial].id;


		}
    current_trial += 1;

    data.pR_red = probabilities[1].shift();
    data.pR_white = probabilities[2].shift();
    data.pR_black = probabilities[3].shift();
    data.outcome = choice_outcomes[key_pressed-1];

  }
}


////////////////////////////////////////////////////////////////////////////////

function  make_learn_phase() {
	console.log('learn phase')
	timeline.push(choice_trial)

	// learn phase, contexts
	// context 1
	// welcome
	jsPsych.addNodeToEndOfTimeline({timeline: [welcome],}, jsPsych.resumeExperiment);
	for (let i = 0; i < first_block_len; i++) {
		jsPsych.addNodeToEndOfTimeline({timeline: [choice_trial,intertrial],}, jsPsych.resumeExperiment);
	}
	jsPsych.addNodeToEndOfTimeline({timeline: [bye,take_break],}, jsPsych.resumeExperiment);


	//travel
	//welcome
	jsPsych.addNodeToEndOfTimeline({timeline: [travel,welcome],}, jsPsych.resumeExperiment);

	// context 2
	for (let i = 0; i < block_len; i++) {
		jsPsych.addNodeToEndOfTimeline({timeline: [choice_trial,intertrial],}, jsPsych.resumeExperiment);
	}
	jsPsych.addNodeToEndOfTimeline({timeline: [bye,take_break],}, jsPsych.resumeExperiment);

	//travel
	//welcome
	jsPsych.addNodeToEndOfTimeline({timeline: [travel, welcome],}, jsPsych.resumeExperiment);

	// context 3
	for (let i = 0; i < block_len; i++) {
		jsPsych.addNodeToEndOfTimeline({timeline: [choice_trial,intertrial],}, jsPsych.resumeExperiment);
	}
	jsPsych.addNodeToEndOfTimeline({timeline: [bye,take_break],}, jsPsych.resumeExperiment);

	// travel
	// welcome
	jsPsych.addNodeToEndOfTimeline({timeline: [travel,welcome],}, jsPsych.resumeExperiment);

	// context 4
	for (let i = 0; i < block_len; i++) {
		jsPsych.addNodeToEndOfTimeline({timeline: [choice_trial,intertrial],}, jsPsych.resumeExperiment);
	}
	jsPsych.addNodeToEndOfTimeline({timeline: [bye,take_break],}, jsPsych.resumeExperiment);


	// travel
	// welcome
	jsPsych.addNodeToEndOfTimeline({timeline: [travel, welcome],}, jsPsych.resumeExperiment);

	//context 5
	for (let i = 0; i < block_len; i++) {
		jsPsych.addNodeToEndOfTimeline({timeline: [choice_trial,intertrial],}, jsPsych.resumeExperiment);
	}
	jsPsych.addNodeToEndOfTimeline({timeline: [bye,take_break],}, jsPsych.resumeExperiment);

	// travel
	// welcome

	jsPsych.addNodeToEndOfTimeline({timeline: [travel, welcome],}, jsPsych.resumeExperiment);

	//context 6
	for (let i = 0; i < block_len; i++) {
		jsPsych.addNodeToEndOfTimeline({timeline: [choice_trial,intertrial],}, jsPsych.resumeExperiment);
	}

	// jsPsych.addNodeToEndOfTimeline({timeline: [bye,all_done,pavlovia_finish],}, jsPsych.resumeExperiment);
  jsPsych.addNodeToEndOfTimeline({timeline: [bye,all_done],}, jsPsych.resumeExperiment);
}


var timeline = []

timeline.push({
	type: 'fullscreen',
	fullscreen_mode: true
});

// timeline.push(pavlovia_init);
timeline.push(instruc1);
timeline.push(move_forward);
timeline.push(instruc2);
timeline.push(move_forward);
timeline.push(instruc3);
timeline.push(move_forward);
timeline.push(instruc4);
timeline.push(move_forward);
timeline.push(instruc5);
timeline.push(blue_win);
timeline.push(blue_alone);
timeline.push(blue_win);
timeline.push(blue_alone);
timeline.push(blue_win);
timeline.push(blue_alone);
timeline.push(blue_no_win);
timeline.push(blue_alone);
timeline.push(blue_win);
timeline.push(blue_alone);
timeline.push(blue_win);
timeline.push(blue_alone);
timeline.push(blue_win);
timeline.push(blue_alone);
timeline.push(blue_no_win);
timeline.push(blue_alone);
timeline.push(blue_win);
timeline.push(blue_alone);
timeline.push(blue_win);
timeline.push(instruc6);
timeline.push(move_forward);
timeline.push(instruc7);



jsPsych.init({
  timeline: timeline,
  display_element: 'jspsych-target',
	preload_images: all_images,
	preload_audio: all_audio,
	use_webaudio: false,
	max_load_time: 600000,
	on_trial_start: function(data) {
		var interaction_data = jsPsych.data.getInteractionData();
		var blur_events = interaction_data.filter({event: 'blur'});
		var focus_events = interaction_data.filter({event: 'focus'});
		var fullscreenenter_events = interaction_data.filter({event: 'fullscreenenter'});
		var fullscreenexit_events = interaction_data.filter({event: 'fullscreenexit'});
		jsPsych.data.get().addToLast({interactions: interaction_data.csv()});
		jsPsych.data.get().addToLast({blur_events: blur_events.csv()});
		jsPsych.data.get().addToLast({focus_events: focus_events.csv()});
		jsPsych.data.get().addToLast({fullscreenenter_events: fullscreenenter_events.csv()});
		jsPsych.data.get().addToLast({fullscreenexit_events: fullscreenexit_events.csv()});},
	on_interaction_data_update: function (data) {
		var interaction_data = jsPsych.data.getInteractionData();
		var blur_events = interaction_data.filter({event: 'blur'});
		var focus_events = interaction_data.filter({event: 'focus'});
		var fullscreenenter_events = interaction_data.filter({event: 'fullscreenenter'});
		var fullscreenexit_events = interaction_data.filter({event: 'fullscreenexit'});
		jsPsych.data.get().addToLast({interactions: interaction_data.csv()});
		jsPsych.data.get().addToLast({blur_events: blur_events.csv()});
		jsPsych.data.get().addToLast({focus_events: focus_events.csv()});
		jsPsych.data.get().addToLast({fullscreenenter_events: fullscreenenter_events.csv()});
		jsPsych.data.get().addToLast({fullscreenexit_events: fullscreenexit_events.csv()});},
	on_close: function (data) {
		var interaction_data = jsPsych.data.getInteractionData();
		var blur_events = interaction_data.filter({event: 'blur'});
		var focus_events = interaction_data.filter({event: 'focus'});
		var fullscreenenter_events = interaction_data.filter({event: 'fullscreenenter'});
		var fullscreenexit_events = interaction_data.filter({event: 'fullscreenexit'});
		jsPsych.data.get().addToLast({interactions: interaction_data.csv()});
		jsPsych.data.get().addToLast({blur_events: blur_events.csv()});
		jsPsych.data.get().addToLast({focus_events: focus_events.csv()});
		jsPsych.data.get().addToLast({fullscreenenter_events: fullscreenenter_events.csv()});
		jsPsych.data.get().addToLast({fullscreenexit_events: fullscreenexit_events.csv()});},
  on_finish: function() {
		var interaction_data = jsPsych.data.getInteractionData();
		var blur_events = interaction_data.filter({event: 'blur'});
		var focus_events = interaction_data.filter({event: 'focus'});
		var fullscreenenter_events = interaction_data.filter({event: 'fullscreenenter'});
		var fullscreenexit_events = interaction_data.filter({event: 'fullscreenexit'});
		jsPsych.data.get().addToLast({interactions: interaction_data.csv()});
		jsPsych.data.get().addToLast({blur_events: blur_events.csv()});
		jsPsych.data.get().addToLast({focus_events: focus_events.csv()});
		jsPsych.data.get().addToLast({fullscreenenter_events: fullscreenenter_events.csv()});
		jsPsych.data.get().addToLast({fullscreenexit_events: fullscreenexit_events.csv()});

    document.body.innerHTML = '<p> <center>Thank you for participating in the first part of the study! Please wait while your data saves. After 10 seconds, you will be redirected to Spark. </center> </p>';
      setTimeout(function () {var end_link = "https://spark.hartleylab.org/completed/".concat(subject_id); window.location = end_link;}, 10000)

}
})
