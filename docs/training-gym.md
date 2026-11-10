# RLVIND: A Training Gym for Training

**Reinforcement Learning with Verifiable Inverse Nonlinear Dynamics**

## The idea

Train a model to make other models better.

1. Start with many cheap toy models, each with a task and a training, evaluation, and inference setup.
2. Ask the model under training to improve them. Reward measured improvement.
3. As the tasks become easy, increase the difficulty and diversity of the target models.
4. Repeat.

This is **the bitter lesson squared**: the target models learn to solve problems, while the outer model learns how to make learning and computation work better. Supply tasks, tools, compute, and verifiable rewards; let learning discover the strategies.

Neural networks are the first gym. The broader hypothesis is that this curriculum can develop transferable competence in understanding and improving nonlinear dynamical systems. Recursive self-improvement is a particularly valuable application because the learner is itself one of those systems.

## Why start with models?

The opportunity is a renewable supply of useful training experience. We can generate architectures, datasets, training runs, checkpoints, and inference problems; ask a learner to improve them; and measure what happens. Each episode can produce a training signal without requiring a human to supply the solution.

Models also provide an unusually convenient experimental domain. We can save their state, clone them, inspect their internals, change their parameters, and run controlled comparisons. For questions about a neural network, the experiment can run the actual network being studied.

The motivating result is *Sparks of Weight-Space Programming*. Its authors report that a frontier model improved Qwen3.5-0.8B by making targeted weight edits that reduced repetitive reasoning. Under their evaluation protocol, MATH-500 accuracy rose from 7.9% to 48.1%. The editing experiment used practice problems and execution feedback, while prohibiting gradient-based training. [1][2]

That establishes a useful starting capability in a limited setting. RLVIND asks what happens when we deliberately train that capability across many environments and keep increasing their difficulty.

The supply of possible environments is enormous. The practical challenge is to keep generating environments that teach something new at an affordable computational cost.

## The training loop

There are two roles:

- **Learner:** the capable model receiving RL updates.
- **Target:** the model or learning process that the learner is asked to improve during an episode.

An episode gives the learner a target, an objective, development data, tools, and a resource budget. The learner investigates, makes changes, and submits a result. The environment evaluates that result on held-out cases and returns a reward.

```text
initialize a capable learner
initialize a population of inexpensive model-improvement tasks

repeat:
    sample target models and objectives
    let the learner attempt improvements within a budget
    evaluate submitted results on held-out cases
    update the learner from the measured rewards
    increase task difficulty and diversity as performance improves
```

The learner can write code, inspect tensors and activations, change training data, adjust optimization, modify architectures, improve inference procedures, or edit weights directly. Weight-space programming expands the available actions; conventional training remains available too.

For a simple episode, the reward is the target's held-out score improvement under a fixed budget. Other episodes can reward lower training cost, faster inference, or better generalization while requiring an agreed level of task performance. Each objective defines what counts as improvement before the episode begins.

The evaluator controls the held-out cases and scoring rule. Charge the learner for the experiments it runs, including unsuccessful attempts. These basic constraints keep the reward connected to useful improvement.

## Progressive overload

Begin with targets cheap enough to support many attempts: small classifiers, sequence models, recurrent networks, and transformers. Include models at different training stages and with different amounts of room for improvement.

As the learner becomes competent, increase the load along several dimensions:

| Dimension | Progression |
| --- | --- |
| Model complexity | Small networks to larger and more varied architectures |
| Task difficulty | Simple synthetic tasks to richer combinations and real workloads |
| Improvement difficulty | Obvious weaknesses to stronger, well-tuned baselines |
| Experimental horizon | Short changes to decisions whose effects appear after longer training or inference |
| Transfer | Familiar setups to unfamiliar model families and data distributions |

Retain a mixture of easier and harder tasks. Expand the curriculum where performance is saturating, and keep enough tractable tasks to provide a useful learning signal.

Coding agents can help construct and maintain this population. The essential requirement is an executable environment with a meaningful objective and a reliable evaluation. The learner's methods can evolve through training.

## The path to recursive self-improvement

Initially, the learner improves small external targets. As its competence grows, the target population can include increasingly capable models, then checkpoints and training procedures related to the learner itself.

At that point, the same activity can produce an improved successor. If the successor is also better at making subsequent improvements, the loop becomes recursive.

The learner need not acquire a complete theory of itself first. A useful training procedure, inference improvement, or parameter change could be enough to advance one generation. Multiple useful discoveries can accumulate.

The central uncertainty is whether strategies learned on simpler targets transfer to increasingly capable ones. Progressive difficulty makes that an empirical question throughout the curriculum. The first milestone is a rising transfer curve; successful recursion is a later milestone.

## Why RLVIND, and where it can go

**Verifiable** describes the training signal: proposed improvements and predictions are checked against execution or observation. Falsifiable explanations may be valuable products of the work, but an episode can have a verifiable outcome without requiring a complete explanatory theory.

**Inverse** describes working backward from observed or desired behavior toward the parameters, mechanisms, or procedures that produce it. Model improvement is an initial form of inverse design. **Nonlinear dynamics** keeps the scope broad enough to include both neural computation and other systems.

The longer-term bet is that the learner acquires reusable ways to investigate unfamiliar dynamics: choosing informative experiments, detecting structure, predicting consequences, and finding effective changes. Those skills could be tested in physical simulations, engineering systems, and eventually experiments involving the physical world.

Cross-domain transfer remains a separate hypothesis. Each new domain needs its own trustworthy observations and evaluation. Neural networks are a practical starting point because they offer accessible internals, repeatable experiments, and a direct connection to improving the learner.

## The first experiment

Start with one capable open-weight learner and a diverse collection of inexpensive target models. Run RL on model-improvement episodes and periodically evaluate the learner on unfamiliar targets, including held-out task and architecture families.

Compare against the starting model under the same experimental budget. An equal-compute continuation on conventional coding and math tasks provides a useful additional baseline.

The main question is:

> As we spend more training compute on progressively harder model-improvement tasks, does the learner become better at improving unfamiliar models?

Measure that learning curve, increase the load when the current tasks saturate, and follow the curve toward larger targets. Cheap targets, reusable checkpoints, and occasional learner updates make this a reasonable starting design for a small number of accelerator nodes; the achievable scale will depend on measured episode and training costs.

**Build the gym. Train on it. Keep increasing the load.**

## References

1. Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/). November 2, 2026.
2. Authors' [Experiment 8 protocol, results, and reproduction details](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md).
