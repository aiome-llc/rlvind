# RLVIND: A Training Gym for Neural Computation

**Reinforcement Learning with Verifiable Inverse Nonlinear Dynamics**

## The idea

Train a model to make other models better.

RLVIND trains agents through experiments on neural computation. Its environments focus on neural network training,
evaluation, and direct weight programming. The immediate objective is transferable competence in improving unfamiliar
neural systems. The longer-term hypotheses are that this competence can support recursive self-improvement and
generalize to understanding and controlling other nonlinear systems, including those in biology, chemistry, and physics.

The starting point is simple:

1. Build many inexpensive target models, each with a task, tools, an evaluation, and a resource budget.
2. Ask a capable learner to improve them. Reward measured results and train the learner with RL.
3. As the tasks become easy, increase the difficulty and diversity of the targets.
4. Repeat, measuring progress on unfamiliar models throughout.

This is **the bitter lesson squared**: the targets learn to solve problems, while the learner learns how to make learning
and computation work better. Supply tasks, tools, compute, and verifiable rewards; let learning discover the strategies.

## Why neural computation comes first

Neural computation provides a renewable supply of useful experiments. We can generate architectures, datasets, training
runs, checkpoints, and inference problems; ask a learner to investigate and improve them; and measure what happens. Each
episode can produce a training signal without requiring a human to supply the solution.

It also provides an unusually controllable laboratory. We can save a network's state, clone it, inspect its internals,
change its parameters, and run controlled comparisons. For a question about a network, the experiment can run the actual
network being studied. Neural systems are both a scientific subject in their own right and our first instrument for
developing more capable experimental agents.

A motivating example is
[*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/). Its authors report
that a frontier model improved Qwen3.5-0.8B through targeted weight edits that reduced repetitive reasoning. MATH-500
accuracy rose from 7.9% to 48.1% under their
[reported evaluation protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md). The experiment
provided practice problems and execution feedback while prohibiting gradient-based training.

This is a starting capability in a limited setting. RLVIND asks what happens when we deliberately train the ability to
investigate and improve neural systems across many environments, then keep increasing their difficulty.

## The gym

There are two roles:

- **Learner:** the capable model receiving RL updates.
- **Target:** the network or learning process that the learner investigates and improves during an episode.

An episode gives the learner a target, an objective, development data, tools, and a resource budget. The learner runs
experiments and submits a result. The environment evaluates that result on held-out cases and returns a reward.

The initial environment families cover complementary parts of this work:

| Family | Example actions | Measured outcome |
| --- | --- | --- |
| Training | Change data selection, optimization, initialization, or architecture | Held-out quality at a fixed training budget, or cost to reach a specified quality |
| Evaluation and generalization | Probe checkpoints, predict failures, select a model or intervention | Prediction accuracy or performance of the selected model on fresh cases |
| Direct weight programming | Inspect tensors and activations, edit weights, or construct a small network | Task performance under the episode's resource and behavior constraints |

The learner can write code, use conventional training, edit weights directly, and change inference procedures. An
environment specifies the available actions. Some tasks can isolate a particular ability; others allow the learner to
combine methods.

For a simple improvement episode, the reward is the target's held-out score gain under a fixed experimental budget.
Other episodes can reward lower training cost, faster inference at a required quality, or accurate predictions about a
target's behavior. Each objective defines what counts as success before the episode begins.

The evaluator controls the scoring rule and held-out cases. All experiments, including unsuccessful attempts, count
toward the learner's budget. This keeps the reward connected to useful results and gives efficient investigation value.

An episode can have a verifiable outcome without a complete explanatory theory. A measured improvement establishes an
effect under the tested conditions; predictions about other conditions can be checked in further experiments.

The [training gym design](training-gym.md) describes the initial tasks and episode protocol.

## Progressive overload

Begin with targets cheap enough to support many attempts: small classifiers, sequence models, recurrent networks, and
transformers. Include different training stages and different amounts of room for improvement, including competent
baselines as well as targets with obvious weaknesses.

As the learner becomes competent, increase the load along several dimensions:

| Dimension | Progression |
| --- | --- |
| Model complexity | Small networks to larger and more varied architectures |
| Task difficulty | Simple synthetic tasks to richer combinations and real workloads |
| Improvement difficulty | Obvious weaknesses to stronger, well-tuned baselines |
| Experimental horizon | Short interventions to decisions whose effects emerge after longer training or inference |
| Transfer | Familiar setups to unfamiliar model families and data distributions |

Retain a mixture of easier and harder tasks. Expand the curriculum where performance is saturating, and keep enough
tractable tasks to provide a useful learning signal. Coding agents can help construct and maintain this population.
The practical challenge is to keep producing environments that teach something new at an affordable computational cost.

## The first experiment

Start with one capable open-weight learner and a diverse collection of inexpensive targets. Run RL on the gym's tasks
and periodically evaluate the learner on unfamiliar targets, including whole task, architecture, and environment
generator families held out from training. Keep this evaluation suite separate from the episodes that provide RL rewards.

Compare against the starting learner under the same experimental budget and against standard optimization or search
methods where applicable. An equal-compute continuation on conventional coding and math tasks provides an additional
baseline for whether the gym is a useful way to spend training compute.

The main question is:

> As we spend more training compute on progressively harder neural computation tasks, does the learner become better at
> improving unfamiliar models?

Measure improvement and experimental cost across target families and sizes. Use repeated runs to distinguish gains from
training and evaluation noise. Track both the compute used to train the learner and the resources it spends within each
episode. The first milestone is a rising transfer curve at a matched experimental budget.

Cheap targets and reusable checkpoints make it possible to begin with a small pilot. Measure episode cost, training
throughput, and reward quality before choosing the scale of a larger run. Increase the load as progress warrants it.

## Two longer-term hypotheses

### Recursive self-improvement

Initially, the learner improves external targets. As its competence grows, the population can include increasingly
capable models, then checkpoints and training procedures related to the learner itself. Useful changes to training,
inference, or weights could produce an improved successor.

The test for recursion is whether the successor is better at producing further improvements under comparable resource
budgets. This requires evaluating its ability to investigate and improve unfamiliar targets, alongside its other
capabilities. Successive gains in that ability would provide evidence for recursive self-improvement.

The central uncertainty is whether methods learned on simpler targets transfer to increasingly capable ones. The gym
makes this a question to measure throughout the curriculum. Useful discoveries can accumulate before the learner has a
complete theory of itself.

### Understanding nonlinear systems

The broader ambition is to develop agents that understand nonlinear dynamical systems through experiments. Neural
computation is the first domain in which to test whether the learner develops useful habits: choosing informative
experiments, detecting structure, predicting consequences, and finding effective interventions.

These abilities can be evaluated within the gym. For example, ask the learner to predict which training regimes a change
will help, then test those predictions on new regimes. Reliable predictions and transferable interventions provide
evidence of competence beyond a score gain on one target. Explanatory models can become useful products of this work.

Transfer to biology, chemistry, and physics is a separate hypothesis. It requires experiments in those domains, with
their own observations, intervention limits, and evaluation. Neural improvement could succeed through methods specific
to neural networks; broader scientific competence must earn its evidence independently. RSI and scientific transfer
can each be investigated on their own merits.

## Why RLVIND

**Verifiable** describes the training signal: improvements and predictions are checked against execution or observation.
The scope of a result is the conditions under which it was evaluated.

**Inverse** describes working backward from observed or desired behavior toward the parameters, mechanisms, and
procedures that produce it. Improving neural computation is an initial form of inverse design.

**Nonlinear dynamics** expresses the broader scientific ambition. The immediate work is to build a gym in which agents
learn to investigate and improve neural systems, and follow the evidence toward larger targets and other domains.
