# Implementation Plan: JAX-based Fine-tuning CLI Script (Causal LM + FSDP & Context Parallel)

This plan details the steps to create a CLI script for fine-tuning Hugging Face pre-trained base LLMs (Causal Language Modeling) using JAX/Flax, with support for Fully Sharded Data Parallel (FSDP) and Context Parallel (Sequence Parallelism).

## 1. Project Initialization
- [ ] Create the main script file `train.py`.
- [ ] Create a `requirements.txt` file listing dependencies:
    - `jax`
    - `jaxlib`
    - `flax`
    - `transformers`
    - `datasets`
    - `optax`
    - `torch` (for data loading)
    - `tqdm` (for progress bars)

## 2. CLI Argument Implementation
- [ ] Implement `parse_args()` function using `argparse`.
- [ ] Define supported arguments:
    - `--model_name_or_path`: (str) Path to pretrained base model.
    - `--dataset_name`: (str) The name of the dataset to use.
    - `--dataset_config_name`: (str) The configuration name of the dataset.
    - `--output_dir`: (str) Where to store the final model.
    - `--batch_size`: (int) Global batch size.
    - `--learning_rate`: (float) Initial learning rate.
    - `--num_train_epochs`: (int) Total number of training epochs.
    - `--max_seq_length`: (int) The maximum total input sequence length (block size).
    - `--seed`: (int) Random seed for initialization.
    - `--fsdp_mesh_dim`: (int) Number of devices to use for FSDP (Data Parallel axis).
    - `--context_mesh_dim`: (int) Number of devices to use for Context Parallel (Sequence axis).

## 3. Distributed Environment Setup
- [ ] Initialize JAX Mesh using `jax.experimental.mesh_utils.create_device_mesh`.
    - Ensure `fsdp_mesh_dim * context_mesh_dim` equals the total number of available devices (or handle accordingly).
- [ ] Define the Mesh with axis names, e.g., `('data', 'context')`.

## 4. Data Loading Pipeline (PyTorch Backend)
- [ ] Initialize the Tokenizer.
- [ ] Load and preprocess dataset (Tokenize, Chunk, Label).
- [ ] Create `torch.utils.data.DataLoader`.
    - Drop last batch to ensure fixed shapes.
    - Batch size passed to loader should be the *Global* batch size (JAX handles the splitting via sharding).

## 5. Model Initialization (Flax) & Sharding
- [ ] Load `AutoConfig`.
- [ ] Define Logical Partitioning Rules (Sharding Constraints):
    - Map model parameters to the 'data' (FSDP) axis for sharding.
    - (Optional) Apply tensor parallelism rules if needed, but primary focus is FSDP.
- [ ] Initialize the model with `FlaxAutoModelForCausalLM` using `jax.default_device(jax.devices('cpu')[0])` to avoid OOM during init, or use lazy initialization.

## 6. Training State and Optimizer
- [ ] Create optimizer (AdamW) and schedule.
- [ ] Initialize `TrainState`.
- [ ] Apply `jax.jit` + `jax.sharding.NamedSharding` to shard the initial `TrainState` (Parameters + OptState) across the Mesh.
    - Parameters should be sharded along the 'data' axis (FSDP).

## 7. Training Loop (JAX)
- [ ] Define `train_step` function:
    - Use `@jax.jit` with `in_shardings` annotations (or `with_sharding_constraint` inside).
    - **Input Sharding**:
        - Batch dimension: Sharded across 'data' axis (FSDP).
        - Sequence dimension: Sharded across 'context' axis (Context Parallel).
    - Computes Causal LM loss and gradients.
    - Updates model parameters.
- [ ] Define `eval_step` function with similar sharding constraints.
- [ ] Implement the epoch loop:
    - Iterate through the training dataloader.
    - Call `train_step` (JAX automatically handles the distributed communication based on sharding).
    - Log metrics.

## 8. Model Saving
- [ ] Collect weights from the mesh (unshard) to CPU.
- [ ] Save the fine-tuned model and tokenizer to `output_dir`.

## 9. Verification
- [ ] Run the script on a single device (mesh dims = 1, 1).
- [ ] Run the script on multiple devices (if available) to verify sharding setup (e.g., fsdp=2, context=1).
