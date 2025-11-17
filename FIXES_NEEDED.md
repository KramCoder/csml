# Required Fixes Based on Error Analysis

## Issues Identified:

1. **Model Size Violation**: 
   - Current: Model has 1,711,376,384 parameters (1.7B)
   - Required: Maximum 380,000,000 parameters (should use 360M model)
   - Error: `ValueError: Model has 1711376384 parameters, which is greater than the maximum allowed 380000000`

2. **Model Selection for Different Tasks**:
   - Data generation: Should use 1.7B model ONLY
   - All other tasks (SFT training, inference, etc.): Should use 360M model

3. **SFT Training**:
   - Currently training (accuracy 0.31-0.36)
   - Should train on rft.json (which is being generated correctly)

## Files That Need Modification:

1. `datagen.py` - Ensure it uses 1.7B model for data generation
2. `sft.py` - Ensure it uses 360M model for training
3. `base_llm.py` or model loading code - Ensure 360M model is loaded for non-datagen tasks
4. Any other files that load models - Need to check model size selection

## Next Steps:

Need access to:
- README.md (for instructions)
- All Python files in the homework directory
- Understanding of how models are currently being loaded/selected
