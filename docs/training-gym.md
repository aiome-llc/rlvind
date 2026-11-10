# RLVIND Training Gym

The [proposal](proposal.md) sets out the research program and its longer-term hypotheses. This document describes the
initial neural computation environments, episode protocol, and evaluation design.

## Episode contract

The **learner** receives RL updates across episodes. A **target** is the network or learning process it investigates
within an episode. Each environment provides the following:

| Component | Contents |
| --- | --- |
| Target | A reproducible model snapshot or training setup, with its task and starting performance |
| Objective | The metric, required behavior, and any quality or cost constraints |
| Development access | Data, probes, and execution feedback available for experimentation |
| Actions | The code, data, architecture, inference settings, or weights the learner may change |
| Budget | Limits on learner inference, target computation, and other experimental resources |
| Submission | An executable change, model artifact, selection, or prediction, as appropriate to the task |
| Evaluation | An environment-controlled procedure that scores the submission on held-out cases |

The objective, constraints, and scoring rule are fixed before the learner begins. Final evaluation data and scoring
remain under the environment's control. A learner's own development tests can inform its decisions; the environment
runs the final evaluation independently.

## Initial environments

Start with a small, varied collection of inexpensive targets. These examples define useful task families; their sizes
and budgets should be chosen from measured episode costs.

### Training

Give the learner a small classifier or sequence model, a dataset, and a training budget. Let it change optimization,
data selection, initialization, or architecture. Score held-out performance under the budget, or the cost of reaching a
specified performance level.

Vary the target's training stage and baseline quality. Include well-tuned starting points as the learner improves, so
that the curriculum continues to demand useful discoveries.

### Evaluation and generalization

Give the learner several checkpoints or candidate interventions and a limited probing budget. Ask it to select the
candidate expected to perform best on a specified task distribution. Evaluate the selection on fresh cases.

Other episodes can ask for predictions of performance or intervention effects. Commit the predictions before the
environment runs the hidden experiments, and score them with a fixed prediction metric. The task specification states
what conditions the prediction covers. These tasks test whether the learner can use experiments to make reliable
decisions and anticipate behavior.

### Direct weight programming

Give the learner a network with an improvement objective, access to its tensors and activations, and development probes.
Let it submit weight changes and evaluate the resulting network on held-out inputs. Some tasks can ask it to construct
a small network for a specified behavior.

Check any required preservation of existing capabilities alongside the desired improvement. Episodes that study direct
editing can restrict conventional training; mixed episodes can allow both and charge their computational costs to the
same experimental budget.

Inference procedures can also be changed when an environment allows them. Objectives can include faster inference at
a required quality or better performance within a fixed inference budget.

## Training loop

```text
initialize a capable learner
initialize a population of inexpensive neural computation tasks
reserve separate evaluation families

repeat:
    sample targets and objectives from the training population
    give each attempt a reproducible starting state and an experimental budget
    let the learner investigate and submit a result
    evaluate submissions on held-out cases for their training episodes
    update the learner with RL from the measured rewards
    periodically measure transfer on the reserved evaluation families
    increase training difficulty and diversity as performance improves
```

Within an episode, the learner can inspect the target, write code, run experiments, and revise its approach. Record its
actions, resource use, and submitted artifact so that results can be reproduced. Restore the starting state for
independent attempts and baseline comparisons.

## Rewards and costs

For improvement tasks, begin with held-out score gain over the starting target under a fixed experimental budget.
For efficiency tasks, reward reduced cost while enforcing the required quality. For selection or prediction tasks,
use the outcome or prediction metric specified by the environment.

Count all attempts against the experimental budget, including failed training runs, discarded edits, and unsuccessful
probes. Track learner inference and target computation separately, with explicit resource limits. Record the cost of
environment generation and final evaluation as part of the overall training cost.

Compare stochastic targets over repeated runs, using matched seeds where appropriate. Evaluate meaningful regressions
as part of the objective, such as a proposed edit improving one behavior while damaging another required behavior.
Successful interventions and accurate predictions supply rewards; explanations can be recorded for later investigation.

## Measuring transfer

Training episodes use held-out cases to produce RL rewards. A separate evaluation suite measures progress and supplies
no RL updates. Reserve evaluation tasks at three levels:

- New instances within familiar task families.
- Unfamiliar task, data, and architecture families.
- Environments from generators or authors whose tasks were excluded from training.

Use validation tasks to guide curriculum changes and keep a final test suite untouched for reporting. This separation
helps distinguish learning to improve networks from adapting to a particular task generator or repeatedly consulted test.

Compare the trained learner with its starting checkpoint under the same episode resources. Include standard training
or search methods where applicable, and a learner given an equal-compute continuation on conventional coding and math
tasks. Report quality gains, prediction accuracy where relevant, and experimental cost by family and target size, with
uncertainty across repeated runs. Plot progress against the compute spent training the learner.

## First run

1. Build a small mixture of training, evaluation, and weight programming tasks. Reproduce baseline outcomes and measure
   episode costs.
2. Run a pilot with one capable open-weight learner. Check whether RL improves results on unfamiliar targets at matched
   experimental budgets.
3. Increase target difficulty and diversity where the learning curves justify it, retaining easier tasks that still
   provide a useful signal.

The first deliverable is a reproducible gym and a measured transfer curve. Larger targets and tasks involving the
learner's own training process follow as the evidence and available compute allow.
