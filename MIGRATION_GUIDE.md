# Migration Guide

## Overview

The MARL repository has been refactored to separate the generic MARL framework from domain-specific electricity market code. This guide helps you migrate from the old structure to the new one.

## What Changed?

### Old Structure
```
easy_marl/
  src/                    # Mixed generic + electricity-specific code
    agents.py
    environment.py        # MARLElectricityMarketEnv
    observators.py
  examples/
    bidding/              # Partial electricity market code
      market.py
      training.py
```

### New Structure
```
easy_marl/
  core/                   # Generic MARL framework
    agents.py             # BaseAgent, PPOAgent, SimpleAgent
    environment.py        # MARLEnvironment (base class)
    training.py           # Generic training loops (IBR/SBR)
  examples/
    electricity_market/   # Complete electricity market example
      environment.py      # ElectricityMarketEnv (extends MARLEnvironment)
      market.py           # Market clearing functions
      observators.py      # Market-specific observers
      params.py           # Parameter generation
      training.py         # Market-specific training wrappers
```

## Migration Steps

### Step 1: Update Imports

**Old imports (deprecated but still work):**
```python
from easy_marl.src.agents import PPOAgent, SimpleAgent
from easy_marl.src.environment import MARLElectricityMarketEnv
from easy_marl.src.observators import OBSERVERS
from easy_marl.examples.bidding.training import auto_train, sequential_train
from easy_marl.examples.bidding.market import market_clearing
```

**New imports (recommended):**
```python
# Generic framework
from easy_marl.core.agents import PPOAgent, SimpleAgent, BaseAgent
from easy_marl.core.environment import MARLEnvironment
from easy_marl.core.training import sequential_train, parallel_train

# Electricity market example
from easy_marl.examples.electricity_market.environment import ElectricityMarketEnv
from easy_marl.examples.electricity_market.observators import OBSERVERS
from easy_marl.examples.electricity_market.training import auto_train, sequential_train
from easy_marl.examples.electricity_market.market import market_clearing
from easy_marl.examples.electricity_market.params import make_default_params
```

### Step 2: Update Class Names (if needed)

**Old:**
```python
from easy_marl.src.environment import MARLElectricityMarketEnv

env = MARLElectricityMarketEnv(agents=agents, params=params)
```

**New:**
```python
from easy_marl.examples.electricity_market.environment import ElectricityMarketEnv

env = ElectricityMarketEnv(agents=agents, params=params)
```

Note: `MARLElectricityMarketEnv` is aliased to `ElectricityMarketEnv` for backward compatibility.

### Step 3: No Code Changes Required!

Thanks to backward compatibility, your existing code will continue to work. You'll see deprecation warnings that guide you to the new imports:

```
DeprecationWarning: easy_marl.src is deprecated. Please use easy_marl.core instead.
DeprecationWarning: easy_marl.examples.bidding is deprecated. Please use easy_marl.examples.electricity_market instead.
```

## Key Improvements

### 1. Fixed PPOAgent Bug

**Issue:** The old `PPOAgent.fixed_act_function()` didn't truly freeze the policy - it changed when the agent was trained.

**Fix:** The new implementation uses serialization to create a true snapshot of the policy.

**Impact:** Training with IBR/SBR now works correctly with proper frozen policies.

### 2. Generic Framework

The new `easy_marl.core` module provides reusable components:

- `BaseAgent`: Abstract base class for all agents
- `MARLEnvironment`: Abstract base class for MARL environments
- `sequential_train()` and `parallel_train()`: Generic training loops

### 3. Extensibility

You can now easily create MARL environments for new domains:

```python
from easy_marl.core.environment import MARLEnvironment
from easy_marl.core.agents import PPOAgent

class MyEnvironment(MARLEnvironment):
    def _build_observation_space(self):
        # Define your observation space
        pass
    
    def _build_action_space(self):
        # Define your action space
        pass
    
    def _get_obs(self, agent_index=None):
        # Build observation for agent
        pass
    
    def step(self, action):
        # Implement your environment logic
        pass
```

See `easy_marl/examples/electricity_market/` for a complete example.

## Breaking Changes

**None!** All old imports work with deprecation warnings. However, we recommend migrating to the new structure for:

1. Better organization and maintainability
2. Access to generic framework features
3. Clearer separation between generic and domain-specific code
4. Future-proofing your code

## Testing Your Migration

Run your existing tests:

```bash
python -m pytest tests/
```

All tests should pass. Deprecation warnings are expected and can be addressed gradually.

## Getting Help

- See [README.md](README.md) for usage examples
- See [CODE_REVIEW.md](CODE_REVIEW.md) for detailed architecture analysis
- Check the electricity market example in `easy_marl/examples/electricity_market/`

## Timeline

- **Now:** Old imports work with deprecation warnings
- **Future:** Old structure may be removed in a major version update (with advance notice)

We recommend migrating at your convenience to take advantage of the improved architecture.
