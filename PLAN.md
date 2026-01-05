# Implementation Plan: JAX-based Fine-tuning CLI Script

This plan details the steps to create a CLI script for fine-tuning Hugging Face pre-trained models using JAX/Flax, utilizing PyTorch for efficient data loading.

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
    - `--model_name_or_path`: (str) Path to pretrained model or model identifier from huggingface.co/models.
    - `--dataset_name`: (str) The name of the dataset to use (via the datasets library).
    - `--output_dir`: (str) Where to store the final model.
    - `--batch_size`: (int) Batch size per device.
    - `--learning_rate`: (float) Initial learning rate.
    - `--num_train_epochs`: (int) Total number of training epochs.
    - `--max_seq_length`: (int) The maximum total input sequence length after tokenization.
    - `--seed`: (int) Random seed for initialization.

## 3. Data Loading Pipeline (PyTorch Backend)
- [ ] Initialize the Tokenizer from `model_name_or_path`.
- [ ] Load dataset using `datasets.load_dataset`.
- [ ] Implement preprocessing function to tokenize texts.
- [ ] Implement `collate_fn` for batching.
- [ ] Create `torch.utils.data.DataLoader` for training and validation sets.
    - Ensure `drop_last=True` for JAX fixed shape compilation compatibility if necessary (or handle variable shapes, but fixed is easier/faster).

## 4. Model Initialization (Flax)
- [ ] Load the configuration with `AutoConfig`.
- [ ] Initialize the model with `FlaxAutoModelForSequenceClassification` (focusing on classification as the primary use case).
- [ ] Initialize the random seed `jax.random.PRNGKey`.

## 5. Training State and Optimizer
- [ ] Create a linear decay schedule with warmup using `optax`.
- [ ] Create the optimizer (e.g., AdamW) using `optax`.
- [ ] Initialize `TrainState` with the model parameters and optimizer.

## 6. Training Loop (JAX)
- [ ] Define `train_step` function:
    - Decorated with `@jax.jit`.
    - Computes loss and gradients.
    - Updates model parameters.
- [ ] Define `eval_step` function:
    - Decorated with `@jax.jit`.
    - Computes metrics (accuracy/loss) on validation batches.
- [ ] Implement the epoch loop:
    - Iterate through the training dataloader.
    - Call `train_step` and log training loss.
    - Run evaluation at the end of each epoch using `eval_step`.
    - Print epoch summary.

## 7. Model Saving
- [ ] Save the fine-tuned model and tokenizer to `output_dir` using `model.save_pretrained` and `tokenizer.save_pretrained`.

## 8. Verification
- [ ] Run the script with a lightweight model (e.g., `distilbert-base-uncased`) and a small dataset (e.g., `glue` task `mrpc`) to ensure the pipeline works correctly.
