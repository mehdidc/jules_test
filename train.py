import argparse
import logging
import os
import sys
from typing import Any, Tuple, Optional

import datasets
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import torch
from flax.training import train_state
from flax.training.common_utils import shard
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    HfArgumentParser,
    is_tensorboard_available,
)
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from flax.serialization import to_bytes, from_bytes

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Global Batch size for the training iterator.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--fsdp_mesh_dim",
        type=int,
        default=1,
        help="Number of devices to use for FSDP (Data Parallel axis).",
    )
    parser.add_argument(
        "--context_mesh_dim",
        type=int,
        default=1,
        help="Number of devices to use for Context Parallel (Sequence axis).",
    )

    args = parser.parse_args()
    return args

def setup_mesh(args):
    """
    Sets up the JAX mesh for distributed training.
    """

    local_device_count = jax.local_device_count()
    global_device_count = jax.device_count()

    logger.info(f"Local devices: {local_device_count}, Global devices: {global_device_count}")

    mesh_shape = (args.fsdp_mesh_dim, args.context_mesh_dim)

    # In a real distributed setting, we would need to check across hosts.
    # Here we perform a basic check against visible devices.

    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    # Create the mesh
    # mesh_utils.create_device_mesh creates a mesh in row-major order.
    # We map this to (data, context)

    # If using more devices than available (e.g. testing), this will error out unless we are careful.
    # Assuming the user provides correct dims for the available hardware.

    try:
        device_mesh_shape = mesh_utils.create_device_mesh(mesh_shape)
    except ValueError as e:
        logger.error(f"Error creating device mesh with shape {mesh_shape}. Available devices: {global_device_count}. Error: {e}")
        raise e

    mesh = Mesh(device_mesh_shape, axis_names=('data', 'context'))

    return mesh

def get_dataset(args, tokenizer):
    """
    Loads and preprocesses the dataset.
    """
    if args.dataset_name is not None:
        dataset = datasets.load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        raise ValueError("Need to specify --dataset_name")

    # Assuming 'train' split exists, otherwise fallback
    if "train" not in dataset:
        raise ValueError("Dataset does not have a 'train' split.")

    # CORRECT APPROACH:
    raw_datasets = datasets.load_dataset(args.dataset_name, args.dataset_config_name)
    if "validation" not in raw_datasets:
        raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1, seed=args.seed)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_seq_length

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets["train"], lm_datasets["test"] if "test" in lm_datasets else lm_datasets["validation"]


def data_collator(features):
    # Standard collator for causal LM
    first = features[0]
    batch = {}

    # Convert to tensor
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch

def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Setup Mesh
    mesh = setup_mesh(args)
    logger.info(f"Initialized Mesh: {mesh}")

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Data Loading
    train_dataset, eval_dataset = get_dataset(args, tokenizer)

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(eval_dataset)}")

    # Create DataLoaders
    # Note: We use drop_last=True to ensure fixed shapes for JAX compilation
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=0 # JAX usually prefers main thread or careful multiprocess
    )

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=0
    )

    # Model Config
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    # Initialize Model
    # We use lazy initialization/empty weights initially to avoid OOM on CPU before sharding
    # But Flax models init on CPU by default.
    # To shard parameters from the start, we can init inside the mesh context or use jax.jit

    logger.info("Initializing model...")
    model = FlaxAutoModelForCausalLM.from_config(config, seed=args.seed)

    # Initialize parameters
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    # We define the sharding rules
    # FSDP means sharding parameters along the 'data' axis.
    # In Flax, we can use logical partitioning.
    # We will shard all parameters that are not scalars along 'data' axis.

    # Simple FSDP-like strategy: Shard the largest dimension of parameters along 'data' axis.
    # Or cleaner: Shard 'embed_embedding', 'kernel', 'bias' etc.
    # For simplicity in this script, we will shard everything fully (FSDP style).
    # Since we don't have NamedSharding embedded in the model definition easily without modification,
    # we can use jax.tree_util.tree_map to define PartitionSpecs.

    # However, to use true FSDP with JAX, we usually map params to P('data').

    # Create TrainState
    def create_train_state(params):
        tx = optax.adamw(learning_rate=args.learning_rate)
        return train_state.TrainState.create(
            apply_fn=model.__call__,
            params=params,
            tx=tx,
        )

    # Init params on CPU first (unless too big, then we need abstract init)
    # model.from_config initializes params in model.params
    params = model.params

    state = create_train_state(params)

    # Define Sharding for State
    # We want to shard the state (params and opt_state) across the 'data' axis.
    # This effectively implements FSDP (sharding parameters across data devices).

    # Rule: Map every leaf array in params to P('data')
    # Use P('data', None) or similar if high rank?
    # FSDP usually shards the first dimension of the weights if they are flattened, or just shards the weights.
    # If we shard weights by 'data', each device holds 1/N of the weight.

    def get_sharding(x):
        # Shard only if rank >= 1
        if hasattr(x, "ndim") and x.ndim >= 1:
            return NamedSharding(mesh, P('data'))
        return NamedSharding(mesh, P())

    state_sharding = jax.tree_util.tree_map(get_sharding, state)

    # Put state on mesh
    state = jax.device_put(state, state_sharding)

    logger.info("Model and Optimizer initialized and sharded.")

    # Training Step
    # We need to define sharding for inputs
    # Batch dim -> 'data', Sequence dim -> 'context'
    # Wait, if we use Context Parallel, we split Sequence dim across 'context' axis.
    # And Batch dim across 'data' axis.

    # Input Sharding: P('data', 'context') for (batch, seq)
    # Output (Loss) Sharding: P() (replicated or scalar)

    @jax.jit
    def train_step(state, batch):

        def loss_fn(params):
            logits = state.apply_fn(
                params=params,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = batch["labels"][..., 1:]

            # Flatten
            # We need to be careful with cross entropy loss in distributed settings if not using appropriate optax/loss functions that support it
            # But here we are computing loss on each device for its shard of data?
            # No, JAX handles this if we use JIT with sharding.

            loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
            return loss.mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)

        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    @jax.jit
    def eval_step(state, batch):
        logits = state.apply_fn(
            params=state.params,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        shift_logits = logits[..., :-1, :]
        shift_labels = batch["labels"][..., 1:]

        loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
        return loss.mean()

    # Loop
    logger.info("Starting training...")

    for epoch in range(args.num_train_epochs):
        # Training
        train_loss = 0.0
        train_steps = 0

        # We need to shard the batch before passing to train_step?
        # Or let JIT handle it via input constraints?
        # JIT with 'in_shardings' is preferred.

        # Define the expected sharding for the batch
        # Batch dict contains tensors of shape (batch, seq_len)
        # We map batch -> 'data', seq -> 'context'
        batch_sharding = NamedSharding(mesh, P('data', 'context'))

        for batch_idx, batch_cpu in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Train")):
            # Convert torch to numpy
            batch_jax = {k: v.numpy() for k, v in batch_cpu.items()}

            # Distribute batch
            # We need to ensure the batch is properly sharded
            # If we just pass numpy array, JAX might replicate or shard depending on default.
            # Using device_put with sharding is explicit.

            sharded_batch = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, batch_sharding), batch_jax
            )

            # Run step
            # Note: We need to re-specify sharding for state to ensure it stays sharded?
            # Ideally state stays on device.
            # But we must tell JIT that the input 'state' is sharded as 'state_sharding'.
            # We can use jax.jit(..., in_shardings=(state_sharding, batch_sharding), out_shardings=(state_sharding, None))
            # But the decorator above didn't specify it.
            # JAX's recent versions infer sharding from inputs if inputs are already jax Arrays on mesh.
            # So `state` is already sharded. `sharded_batch` is already sharded.
            # So standard jax.jit should work and preserve sharding propagation.

            state, loss = train_step(state, sharded_batch)
            train_loss += loss.item() # This will fetch to host
            train_steps += 1

            if batch_idx % 10 == 0:
                logger.info(f"Step {batch_idx}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / train_steps
        logger.info(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Epoch {epoch+1} Average Train Perplexity: {jnp.exp(avg_train_loss):.4f}")

        # Evaluation
        eval_loss = 0.0
        eval_steps = 0
        for batch_cpu in tqdm(eval_loader, desc=f"Epoch {epoch+1} Eval"):
            batch_jax = {k: v.numpy() for k, v in batch_cpu.items()}
            sharded_batch = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, batch_sharding), batch_jax
            )
            loss = eval_step(state, sharded_batch)
            eval_loss += loss.item()
            eval_steps += 1

        avg_eval_loss = eval_loss / eval_steps
        logger.info(f"Epoch {epoch+1} Validation Loss: {avg_eval_loss:.4f}")
        logger.info(f"Epoch {epoch+1} Validation Perplexity: {jnp.exp(avg_eval_loss):.4f}")

    # Saving
    logger.info("Saving model...")
    # Move params to CPU
    # We use fully_replicated sharding for saving
    replicated_sharding = NamedSharding(mesh, P())
    # Or just jax.device_get(state.params) which fetches to host numpy

    params_cpu = jax.device_get(state.params)

    # Create the model directory
    model.save_pretrained(args.output_dir, params=params_cpu)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Model saved.")

if __name__ == "__main__":
    main()
