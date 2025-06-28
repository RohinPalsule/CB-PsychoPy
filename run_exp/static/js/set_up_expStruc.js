'use strict';

function numberRange (start, end) {
  return new Array(end - start).fill().map((d, i) => i + start);
}

function getRandom(arr, n) {
    var result = new Array(n),
        len = arr.length,
        taken = new Array(len);
    if (n > len)
        throw new RangeError("getRandom: more elements taken than available");
    while (n--) {
        var x = Math.floor(Math.random() * len);
        result[n] = arr[x in taken ? taken[x] : x];
        taken[x] = --len in taken ? taken[len] : len;
    }
    return result;
}

var context_offset = 10;
var sample_window = 10; // first sample_window trials of each context are available for later memory probes
var num_blocks = 6;
var num_contexts = 6;
var first_block_len = 30; // added 10.25.21
var block_len = 40;
var ctx_bump = 2;

var num_bandits = 3;
var init_payoff = [60,30,10];
var decayTheta = init_payoff;
var payoff_bounds = [5,95];
var decay_lambda = 0.6;
var drift_sigma = 8;
var drift_noise = 8;

var num_trials = block_len * num_blocks * 2;
var num_probes = sample_window * (num_blocks-1);
var num_invalid_probes = Math.round(num_probes/5);
num_probes = num_probes + num_invalid_probes;
var mean_ct = 5;
var max_ct = 8;
var min_ct = 2;

var log_rand = Array.apply(null, {length: num_probes}).map(x=>Math.random()).map(Math.log);
var log_rand_div_mean_ct = log_rand.map(function(x) {return x/(1/mean_ct)});
var choice_blocks = log_rand_div_mean_ct.map(Math.ceil).map(function(x){return (x*-1) + min_ct});
choice_blocks = choice_blocks.map(function(x) {if (x > max_ct) {return max_ct} else {return x}});


const isAboveThreshold = (currentValue) => currentValue > max_ct;
const isBelowThreshold  = (currentValue) => currentValue < min_ct;
const sum_mer = (accumulator, curr) => accumulator + curr;
const cumulativeSum = (sum => value => sum += value)(0);
const mean = (array) => array.reduce((a, b) => a + b) / array.length;

var rotation_trials = [40,80,120,160,200,240,280,320,360,400,440];
var deterministic_trials = [30,70,110,150,190];

for (let i = 0; i < num_contexts; i++) {
  for (let j = 1; j < ctx_bump; j++) {
    var trial_to_add = deterministic_trials[i] + j;
    deterministic_trials.push(trial_to_add)
  }
}

while ((choice_blocks.reduce(sum_mer) != (block_len * num_blocks)) | (choice_blocks.some(isBelowThreshold)) | (choice_blocks.some(isAboveThreshold))) {
  // 1. randomly select index
  var ind = parseInt(Math.ceil(Math.random()*num_probes),10) - 1;
  // 2. make equal to num_trials/2
  choice_blocks[ind] = choice_blocks[ind] - Math.sign(choice_blocks.reduce(sum_mer) - (block_len * num_blocks));
  // 3. ensure no element goes below threshold
  choice_blocks = choice_blocks.map(function(x) {if (x < min_ct) {return min_ct} else {return x}});
  // 4. ensure no element goes above threshold
  choice_blocks = choice_blocks.map(function(x) {if (x > max_ct) {return max_ct} else {return x}});
}

choice_blocks = choice_blocks.map(function (x) {return x-1})
choice_blocks[0] = choice_blocks[0] + 10 // added 10.25.21

var mem_probe_trials = choice_blocks.map(function (x) {return x+1}).map(cumulativeSum).map(function (x) {return x+(block_len*num_blocks)})
mem_probe_trials = mem_probe_trials.map(function (x) {return x-1}) // zero-indexing

var trial_nums = numberRange(0,num_trials)
var choice_trials = trial_nums.filter(function(x) { return mem_probe_trials.indexOf(x) < 0 }) // set difference

// randomize the list of memory probe indexes
mem_probe_trials = mem_probe_trials
  .map((value) => ({ value, sort: Math.random() }))
  .sort((a, b) => a.sort - b.sort)
  .map(({ value }) => value)

// Now take the first numInvalidProbes # of indexes.
// These will be the invalid probes (novel images).
var invalidProbeTrials  = getRandom(mem_probe_trials,num_invalid_probes)

// Now first sampleWindow trials of each context are available for later memory probes

var availableForMemProbe = [];
for (let bIdx = 1; bIdx < num_blocks; bIdx++) {
  for (let elem = block_len*bIdx; elem < ((block_len*bIdx)+sample_window); elem++) {
    availableForMemProbe.push(elem-10)
  }
}

// randomize the list of memory probe indexes
availableForMemProbe = availableForMemProbe
  .map((value) => ({ value, sort: Math.random() }))
  .sort((a, b) => a.sort - b.sort)
  .map(({ value }) => value)


var payout = new Array(num_bandits);

for (var i = 0; i < num_bandits; i++) {
  payout[i] = new Array(num_trials);
  payout[i][0] = init_payoff[i];
}

var drift = function(trial_idx) {

  if (rotation_trials.includes(trial_idx)) {
    var bestOpt = decayTheta.indexOf(Math.max(...decayTheta)); // get index (bandit) of max value
    while (decayTheta[bestOpt] == Math.max(...decayTheta)) { // while the index (bandit) of the max value is the same shuffle list --> change highest paying
      decayTheta = decayTheta
        .map((value) => ({ value, sort: Math.random() }))
        .sort((a, b) => a.sort - b.sort)
        .map(({ value }) => value)
    }
  }

  // if (trial_idx > ((num_trials/2)-1)) {  // second half of the experiment
  //   // set the average payoff of each bandit to the mean of the bandits in the first half
  //   decayTheta = Array.apply(null, {length: num_bandits}).map(x=>mean(decayTheta))
  // }

  if (trial_idx > 0) {  // after the first one do the drift

    // slower transitions for the first 2 trials
    if (trial_idx%block_len < 2) {
      var decay_lambda_eff = 0.95;
    } else {
      var decay_lambda_eff = decay_lambda;
    }

    for (let this_bandit = 0; this_bandit < num_bandits; this_bandit++) {
      payout[this_bandit][trial_idx] = decay_lambda_eff * payout[this_bandit][trial_idx-1] + (1-decay_lambda_eff)*decayTheta[this_bandit] + normalRandom()*drift_noise;

      // Reflect at specified boundaries [lower, upper]
      if (payout[this_bandit][trial_idx] > payoff_bounds[1]) { // if above then move it back in within bounds
          payout[this_bandit][trial_idx] = payoff_bounds[1] - (payout[this_bandit][trial_idx] - payoff_bounds[1]);
      }

      if (payout[this_bandit][trial_idx] < payoff_bounds[0]) {
          payout[this_bandit][trial_idx] = payoff_bounds[0] + (payoff_bounds[0] - payout[this_bandit][trial_idx]);
      };
    };
  };

  // % If we're in the first ctxBump trials of a context, make the best option deterministic
  if (deterministic_trials.includes(trial_idx)) {
    var bestOpt = decayTheta.indexOf(Math.max(...decayTheta));
    payout[bestOpt][trial_idx] = 100
  }
};

// create the timeline for the experiment
for (let trial_idx = 0; trial_idx < num_trials; trial_idx++) {
  drift(trial_idx)
};

console.log("choice_blocks")
console.log(choice_blocks)

console.log("choice_trials")
console.log(choice_trials)

console.log("mem probe trials")
console.log(mem_probe_trials)

console.log("invalid")
console.log(invalidProbeTrials)

console.log("available2")
console.log(availableForMemProbe)

// console.log("contexts")
// console.log(contexts)

console.log("payout")
console.log(payout)
