# grounded-pilot-20260811-v3 blueprint review

Status: **blueprint-approved**. This blueprint has not run Llama.

## Validation summary

- Intents: 24
- Questions per intent: 1
- Difficulty inventory: 15 introductory, 9 intermediate
- All automated blueprint checks: passed

## Pilot reference readiness

| Skill | Approved | Domains | Ready |
|---|---:|---|---|
| AI-FND-01 | 4 | cs50.harvard.edu, inst.eecs.berkeley.edu, ocw.mit.edu, plato.stanford.edu | yes |
| AI-AGT-01 | 2 | cs50.harvard.edu, inst.eecs.berkeley.edu | yes |
| AI-SRC-01 | 3 | cs50.harvard.edu, inst.eecs.berkeley.edu | yes |
| AI-SRC-02 | 2 | inst.eecs.berkeley.edu | yes |
| AI-SRC-03 | 3 | cs50.harvard.edu, inst.eecs.berkeley.edu | yes |
| AI-SRC-08 | 2 | inst.eecs.berkeley.edu, www.redblobgames.com | yes |

## AI-SRC-03 course and solver convention

- The frontier contains generated search nodes that are waiting to be selected for expansion.
- A state enters the solver's reached structure when its node is admitted to the frontier; reached therefore means discovered in these questions.
- Expansion removes the selected node, applies its available actions through the result model, and generates child nodes.
- The explored or expanded set records a state after it is removed from the frontier for expansion; it is not a silent synonym for reached.
- A repeated state is skipped when the existing record is already at least as good.
- For an unexpanded state, a newly discovered lower path cost replaces the stored frontier cost and parent record.
- The current solver does not reopen a state after it enters the expanded set.
- Questions stay at the level of frontier, reached-state, duplicate, and expansion mechanics and never request a BFS, DFS, UCS, Greedy, or A-star expansion trace.

## Question intents

### AI-FND-01-INT-04 — intermediate

- Skill: AI-FND-01
- Canonical objective: Define artificial intelligence and recognise tasks that require intelligent behaviour.
- Assessment focus: Compare two systems and identify which one demonstrates a supported intelligent-system capability.
- Cognitive demand: Compare a short pair of scenarios and apply the approved capability criteria.
- Archetype: paired-system capability comparison
- Preferred references: AI-FND-01-8bbbddaf2aa6, AI-FND-01-b50c85fa00a5
- Required characteristics: Use two concise systems with one clearly requiring problem solving, reasoning, decision making, or learning.; Require comparison rather than definition recall.
- Prohibited ambiguity: Do not imply that automation alone establishes intelligence.; Do not make both systems use a supported intelligent capability.
- Misconception/distractor strategy: Distractors should mistake fixed automation, storage, or arithmetic for intelligent behaviour.

### AI-FND-01-INT-05 — intermediate

- Skill: AI-FND-01
- Canonical objective: Define artificial intelligence and recognise tasks that require intelligent behaviour.
- Assessment focus: Diagnose why a real-world task is an AI application rather than merely a fixed automated operation.
- Cognitive demand: Interpret a scenario and identify the supported intelligent capability that changes the classification.
- Archetype: application-versus-automation diagnosis
- Preferred references: AI-FND-01-d03d77e0aca2, AI-FND-01-8bbbddaf2aa6
- Required characteristics: Use a sourced application category such as speech processing, face recognition, game playing, or route finding.; Contrast it with a fixed operation without adding unsupported implementation claims.
- Prohibited ambiguity: Do not ask whether every instance of the broad technology is AI.; Do not rely on adaptation or autonomy unless stated in the scenario.
- Misconception/distractor strategy: Distractors should focus on superficial hardware, speed, or automation instead of the intelligent task.

### AI-FND-01-INT-06 — intermediate

- Skill: AI-FND-01
- Canonical objective: Define artificial intelligence and recognise tasks that require intelligent behaviour.
- Assessment focus: Apply the intelligent-agent view of AI to connect environmental percepts with selected behaviour.
- Cognitive demand: Interpret a percept-to-action scenario and diagnose the claim that best matches the agent definition.
- Archetype: agent-view scenario interpretation
- Preferred references: AI-FND-01-f7c5eb1ccf76, AI-FND-01-d03d77e0aca2
- Required characteristics: State an environmental input and a resulting action or behaviour.; Assess the relationship without requiring formal tuple notation.
- Prohibited ambiguity: Do not leave the environmental input unstated.; Do not make multiple actions equally compatible with the percept.
- Misconception/distractor strategy: Distractors should reverse input and output or claim that action selection ignores percepts.

### AI-AGT-01-INT-01 — introductory

- Skill: AI-AGT-01
- Canonical objective: Explain how an agent interacts with its environment through sensors and actuators.
- Assessment focus: Recognize that an agent perceives its environment and acts upon it.
- Cognitive demand: Recognize the direction of information and action in a simple interaction.
- Archetype: agent-environment relationship recognition
- Preferred references: AI-AGT-01-31af4c58c9a1
- Required characteristics: Use a one-step agent/environment interaction.; Make perception and action both visible.
- Prohibited ambiguity: Do not ask for a formal agent-function definition.; Do not omit either the input or action side.
- Misconception/distractor strategy: Distractors should reverse who perceives or who acts.

### AI-AGT-01-INT-02 — introductory

- Skill: AI-AGT-01
- Canonical objective: Explain how an agent interacts with its environment through sensors and actuators.
- Assessment focus: Distinguish a sensor input from an actuator output.
- Cognitive demand: Interpret one device role in a simple agent scenario.
- Archetype: sensor-versus-actuator classification
- Preferred references: AI-AGT-01-8efa49112c14
- Required characteristics: Name one clearly perceptive component and one clearly acting component.; Ask for one role classification, not a list.
- Prohibited ambiguity: Do not use a component that plausibly senses and acts at once.; Do not equate a sensor with the percept itself.
- Misconception/distractor strategy: Distractors should swap sensor and actuator roles or confuse them with the environment.

### AI-AGT-01-INT-03 — introductory

- Skill: AI-AGT-01
- Canonical objective: Explain how an agent interacts with its environment through sensors and actuators.
- Assessment focus: Interpret the four PEAS roles in a stated task environment.
- Cognitive demand: Recognize which stated element fills one PEAS role.
- Archetype: single-role PEAS interpretation
- Preferred references: AI-AGT-01-8efa49112c14
- Required characteristics: Ask about one of performance measure, environment, actuators, or sensors in context.; Supply enough context to make the role unique.
- Prohibited ambiguity: Do not require recalling the PEAS expansion as an isolated list.; Do not assign one scenario element to multiple roles.
- Misconception/distractor strategy: Distractors should use the other PEAS roles with parallel wording.

### AI-AGT-01-INT-04 — intermediate

- Skill: AI-AGT-01
- Canonical objective: Explain how an agent interacts with its environment through sensors and actuators.
- Assessment focus: Map sensors and actuators in a navigator-agent scenario and explain their interaction.
- Cognitive demand: Apply both component roles to a short scenario.
- Archetype: two-role navigator application
- Preferred references: AI-AGT-01-31af4c58c9a1, AI-AGT-01-8efa49112c14
- Required characteristics: Use the supported navigator or car-agent context.; Require a correct sensor/actuator pairing.
- Prohibited ambiguity: Do not assume a specific vehicle sensor absent from the scenario.; Do not make multiple pairings technically plausible.
- Misconception/distractor strategy: Distractors should swap sensing and acting or substitute the performance measure.

### AI-AGT-01-INT-05 — intermediate

- Skill: AI-AGT-01
- Canonical objective: Explain how an agent interacts with its environment through sensors and actuators.
- Assessment focus: Explain how limited sensor information makes an environment partially observable.
- Cognitive demand: Diagnose observability from the information available to the agent.
- Archetype: observability scenario diagnosis
- Preferred references: AI-AGT-01-8efa49112c14
- Required characteristics: State what the agent can and cannot observe.; Connect missing state information to partial observability.
- Prohibited ambiguity: Do not infer hidden information not stated in the scenario.; Do not equate partial observability with actuator failure.
- Misconception/distractor strategy: Distractors should confuse incomplete information with poor performance or limited actions.

### AI-AGT-01-INT-06 — intermediate

- Skill: AI-AGT-01
- Canonical objective: Explain how an agent interacts with its environment through sensors and actuators.
- Assessment focus: Diagnose an incorrect PEAS description that assigns an element to the wrong role.
- Cognitive demand: Compare component functions and repair one role error.
- Archetype: PEAS misconception repair
- Preferred references: AI-AGT-01-8efa49112c14
- Required characteristics: Present one explicit misclassification and ask for the correction.; Keep the remaining PEAS roles internally consistent.
- Prohibited ambiguity: Do not include two independent errors.; Do not use a multifunction component with unclear role.
- Misconception/distractor strategy: Distractors should preserve the role swap or confuse the performance measure with an action.

### AI-SRC-01-INT-11 — introductory

- Skill: AI-SRC-01
- Canonical objective: Identify the initial state, actions, transition model, goal test and path cost of a search problem.
- Assessment focus: Identify the initial state in a simple search-problem description.
- Cognitive demand: Recognize the configuration before any action occurs.
- Archetype: initial-state recognition
- Preferred references: AI-SRC-01-4024dce75930, AI-SRC-01-9ba6548d4450
- Required characteristics: State a simple starting configuration and one goal.; Ask only which element is the initial state.
- Prohibited ambiguity: Do not present initial state and start state as separate components.; Do not leave the starting configuration implicit.
- Misconception/distractor strategy: Distractors should use an action, a resulting state, or the goal condition.

### AI-SRC-01-INT-12 — introductory

- Skill: AI-SRC-01
- Canonical objective: Identify the initial state, actions, transition model, goal test and path cost of a search problem.
- Assessment focus: Distinguish an available action from the transition model that gives its resulting state.
- Cognitive demand: Interpret a one-step action/result pair.
- Archetype: action-transition distinction
- Preferred references: AI-SRC-01-4024dce75930, AI-SRC-01-9ba6548d4450
- Required characteristics: Name one available choice and the state produced by taking it.; Ask which description is the action or transition model.
- Prohibited ambiguity: Do not use action and result interchangeably.; Do not require domain knowledge beyond the stated mapping.
- Misconception/distractor strategy: Distractors should swap the action with its result or with the goal test.

### AI-SRC-01-INT-13 — introductory

- Skill: AI-SRC-01
- Canonical objective: Identify the initial state, actions, transition model, goal test and path cost of a search problem.
- Assessment focus: Distinguish a goal test from accumulated path cost in a simple formulation.
- Cognitive demand: Recognize whether a stated rule tests success or measures a path.
- Archetype: goal-test versus path-cost interpretation
- Preferred references: AI-SRC-01-4024dce75930, AI-SRC-01-9ba6548d4450
- Required characteristics: State one success condition and one accumulated route measure.; Ask for a single component classification.
- Prohibited ambiguity: Do not call an individual action cost the whole path cost.; Do not make the success condition depend on an unstated optimum.
- Misconception/distractor strategy: Distractors should confuse reaching the goal with minimizing cost or use one edge cost.

### AI-SRC-02-INT-01 — introductory

- Skill: AI-SRC-02
- Canonical objective: Distinguish between a state space and a search tree.
- Assessment focus: Recognize that a state-space graph represents each state once while a search-tree node represents a path to a state.
- Cognitive demand: Interpret the representational difference in one statement.
- Archetype: representation distinction
- Preferred references: AI-SRC-02-aa97b7fb3bd9
- Required characteristics: Contrast state identity with path-specific node identity.; Use plain language rather than graph-theory formalism.
- Prohibited ambiguity: Do not claim every state appears only once in a search tree.; Do not use tree and graph as generic shape labels.
- Misconception/distractor strategy: Distractors should claim that both structures store only states or only paths.

### AI-SRC-02-INT-02 — introductory

- Skill: AI-SRC-02
- Canonical objective: Distinguish between a state space and a search tree.
- Assessment focus: Explain why the same state can appear in multiple search-tree nodes.
- Cognitive demand: Recognize that different paths create distinct tree nodes.
- Archetype: repeated-state explanation
- Preferred references: AI-SRC-02-aa97b7fb3bd9
- Required characteristics: State that two paths end at the same state.; Ask why the corresponding nodes remain distinct.
- Prohibited ambiguity: Do not require a cycle for the repeated state.; Do not imply that the underlying state-space state is duplicated.
- Misconception/distractor strategy: Distractors should merge nodes by final state or claim that repeated states are impossible.

### AI-SRC-02-INT-03 — introductory

- Skill: AI-SRC-02
- Canonical objective: Distinguish between a state space and a search tree.
- Assessment focus: Classify a root-to-descendant action sequence as a plan represented in the search tree.
- Cognitive demand: Interpret what one displayed root-to-node sequence encodes.
- Archetype: root-to-node plan interpretation
- Preferred references: AI-SRC-02-aa97b7fb3bd9
- Required characteristics: Give a short ordered path from the root to one descendant.; Ask what the sequence represents.
- Prohibited ambiguity: Do not ask for algorithm traversal order.; Do not describe the sequence as an unordered state set.
- Misconception/distractor strategy: Distractors should report only the terminal state, reorder actions, or call it the whole state space.

### AI-SRC-03-INT-01 — introductory

- Skill: AI-SRC-03
- Canonical objective: Explain the roles of the frontier, reached-state set and node expansion.
- Assessment focus: Recognize the frontier as the collection of generated search nodes waiting for expansion.
- Cognitive demand: Interpret the role of one search-process structure.
- Archetype: frontier-role recognition
- Preferred references: AI-SRC-03-0ccb1b224a85, AI-SRC-03-e7ed23f73259
- Required characteristics: Describe generated but not yet expanded nodes.; Keep the question independent of any named search algorithm.
- Prohibited ambiguity: Do not describe the frontier as all states in the state space.; Do not equate the frontier with the expanded set.
- Misconception/distractor strategy: Distractors should describe the reached set, goal test, or complete search tree.

### AI-SRC-03-INT-02 — introductory

- Skill: AI-SRC-03
- Canonical objective: Explain the roles of the frontier, reached-state set and node expansion.
- Assessment focus: Interpret expansion as applying available actions to a selected node and generating child nodes.
- Cognitive demand: Recognize the input and output of one expansion step.
- Archetype: expansion-step interpretation
- Preferred references: AI-SRC-03-0ccb1b224a85
- Required characteristics: Name a selected node and the actions available from its state.; Ask what expansion produces.
- Prohibited ambiguity: Do not equate expansion with merely removing a node.; Do not imply that expansion guarantees a goal.
- Misconception/distractor strategy: Distractors should confuse expansion with goal testing, path reconstruction, or frontier ordering.

### AI-SRC-03-INT-03 — introductory

- Skill: AI-SRC-03
- Canonical objective: Explain the roles of the frontier, reached-state set and node expansion.
- Assessment focus: Distinguish when a state enters reached from when it enters explored or expanded under the project convention.
- Cognitive demand: Interpret two named lifecycle events without tracing an algorithm.
- Archetype: reached-versus-expanded timing distinction
- Preferred references: AI-SRC-03-e7ed23f73259, AI-SRC-03-d4bae55f5e66
- Required characteristics: State explicitly that reached means discovered in the scenario.; Identify frontier admission and later expansion as distinct events.
- Prohibited ambiguity: Do not use reached and explored as unexplained synonyms.; Do not ask about a source convention without naming it.
- Misconception/distractor strategy: Distractors should swap discovery and expansion timing or record the state only at goal testing.

### AI-SRC-03-INT-04 — intermediate

- Skill: AI-SRC-03
- Canonical objective: Explain the roles of the frontier, reached-state set and node expansion.
- Assessment focus: Apply reached-state membership to prevent duplicate frontier insertion in a short graph-search scenario.
- Cognitive demand: Diagnose a repeated-state discovery and choose the correct handling step.
- Archetype: duplicate-discovery scenario
- Preferred references: AI-SRC-03-d4bae55f5e66, AI-SRC-03-e7ed23f73259
- Required characteristics: State that the repeated state's existing record is already at least as good.; Ask whether a second frontier record should be added.
- Prohibited ambiguity: Do not omit whether the state was previously reached.; Do not introduce an unstated lower path cost.
- Misconception/distractor strategy: Distractors should add every generated child, remove the original record, or confuse node paths with state identity.

### AI-SRC-03-INT-05 — intermediate

- Skill: AI-SRC-03
- Canonical objective: Explain the roles of the frontier, reached-state set and node expansion.
- Assessment focus: Apply the cheaper-path rule to update an unexpanded state's stored frontier cost and parent.
- Cognitive demand: Compare two path costs to the same unexpanded state and select the correct record update.
- Archetype: lower-cost frontier-record update
- Preferred references: AI-SRC-03-d4bae55f5e66, AI-SRC-03-0ccb1b224a85
- Required characteristics: Give an existing and newly discovered path cost to the same unexpanded state.; Require updating both the best cost and parent when the new cost is lower.
- Prohibited ambiguity: Do not leave expansion status unstated.; Do not imply that the current solver reopens an expanded state.
- Misconception/distractor strategy: Distractors should keep the higher cost, store duplicate records, or reopen an expanded state.

### AI-SRC-03-INT-06 — intermediate

- Skill: AI-SRC-03
- Canonical objective: Explain the roles of the frontier, reached-state set and node expansion.
- Assessment focus: Diagnose a search-process description that silently mixes reached-on-discovery with explored-after-expansion.
- Cognitive demand: Compare lifecycle claims and repair the single convention error.
- Archetype: search-structure convention diagnosis
- Preferred references: AI-SRC-03-e7ed23f73259, AI-SRC-03-d4bae55f5e66, AI-SRC-03-0ccb1b224a85
- Required characteristics: Name reached as discovery and explored as post-removal expansion in the scenario.; Include exactly one timing or role error to diagnose.
- Prohibited ambiguity: Do not use an unnamed textbook convention.; Do not combine a convention error with an algorithm-ordering error.
- Misconception/distractor strategy: Distractors should preserve the reached/explored swap or confuse expansion with frontier selection.

### AI-SRC-08-INT-01 — introductory

- Skill: AI-SRC-08
- Canonical objective: Explain how a heuristic estimates the remaining cost from a state to the goal.
- Assessment focus: Recognize a heuristic as an estimate of remaining cost from the current state to a goal.
- Cognitive demand: Interpret the direction and estimated nature of h(n).
- Archetype: heuristic-role recognition
- Preferred references: AI-SRC-08-a366da363e17
- Required characteristics: Name a current state and goal.; Ask what the heuristic value represents.
- Prohibited ambiguity: Do not call the estimate exact.; Do not omit whether cost is measured from the start or toward the goal.
- Misconception/distractor strategy: Distractors should describe accumulated start cost, edge cost, or goal-test truth.

### AI-SRC-08-INT-02 — introductory

- Skill: AI-SRC-08
- Canonical objective: Explain how a heuristic estimates the remaining cost from a state to the goal.
- Assessment focus: Distinguish the forward heuristic estimate h(n) from accumulated path cost g(n).
- Cognitive demand: Interpret two supplied cost labels in one state description.
- Archetype: g-versus-h interpretation
- Preferred references: AI-SRC-08-a366da363e17
- Required characteristics: State both an accumulated start cost and an estimated remaining cost.; Ask which value is the heuristic.
- Prohibited ambiguity: Do not require calculating f(n).; Do not label both values as estimates.
- Misconception/distractor strategy: Distractors should swap g and h or add them despite the question asking only for h.

### AI-SRC-08-INT-03 — introductory

- Skill: AI-SRC-08
- Canonical objective: Explain how a heuristic estimates the remaining cost from a state to the goal.
- Assessment focus: Interpret Manhattan distance as a simple heuristic for four-way movement on a square grid.
- Cognitive demand: Perform one small Manhattan-distance calculation and interpret it as an estimate.
- Archetype: one-step grid heuristic application
- Preferred references: AI-SRC-08-cbd77b22bcb9
- Required characteristics: Specify a square grid with four-way movement and two coordinates.; Require one addition of horizontal and vertical differences.
- Prohibited ambiguity: Do not allow diagonal movement.; Do not add path cost already travelled.
- Misconception/distractor strategy: Distractors should use one coordinate difference, diagonal distance, or include g(n).

## Proposed deterministic first-attempt seeds

Base seed: `20260811`

| Intent | Question index | Seed |
|---|---:|---:|
| AI-FND-01-INT-04 | 0 | 1624431487 |
| AI-FND-01-INT-05 | 1 | 3751862752 |
| AI-FND-01-INT-06 | 2 | 965594951 |
| AI-AGT-01-INT-01 | 0 | 3421874111 |
| AI-AGT-01-INT-02 | 1 | 288316075 |
| AI-AGT-01-INT-03 | 2 | 2156472931 |
| AI-AGT-01-INT-04 | 3 | 1510062202 |
| AI-AGT-01-INT-05 | 4 | 1239544048 |
| AI-AGT-01-INT-06 | 5 | 4264789084 |
| AI-SRC-01-INT-11 | 0 | 4129721347 |
| AI-SRC-01-INT-12 | 1 | 3968689575 |
| AI-SRC-01-INT-13 | 2 | 1982436537 |
| AI-SRC-02-INT-01 | 0 | 3641971561 |
| AI-SRC-02-INT-02 | 1 | 2835183362 |
| AI-SRC-02-INT-03 | 2 | 1584601142 |
| AI-SRC-03-INT-01 | 0 | 2851759830 |
| AI-SRC-03-INT-02 | 1 | 2170737470 |
| AI-SRC-03-INT-03 | 2 | 671124761 |
| AI-SRC-03-INT-04 | 3 | 1537508959 |
| AI-SRC-03-INT-05 | 4 | 1696686317 |
| AI-SRC-03-INT-06 | 5 | 2078111574 |
| AI-SRC-08-INT-01 | 0 | 3101149788 |
| AI-SRC-08-INT-02 | 1 | 1319962358 |
| AI-SRC-08-INT-03 | 2 | 2138054861 |

Retry seeds remain deterministic because the attempt index is included in seed derivation.

## Exact Kaggle Llama generation command

The blueprint is approved. Run from a clean, committed worktree with HF_TOKEN configured.

```bash
export MODEL_REPOSITORY="meta-llama/Llama-3.1-8B-Instruct"

python -m scripts.generate_grounded_batch \
  --batch-id grounded-pilot-20260811-v3 \
  --skill-id AI-FND-01 \
  --skill-id AI-AGT-01 \
  --skill-id AI-SRC-01 \
  --skill-id AI-SRC-02 \
  --skill-id AI-SRC-03 \
  --skill-id AI-SRC-08 \
  --all-blueprint-intents \
  --base-seed 20260811 \
  --output outputs/grounded-pilot-20260811-v3 \
  --model-id "$MODEL_REPOSITORY" \
  --prompt-version v3.3 \
  --difficulty mixed
```
