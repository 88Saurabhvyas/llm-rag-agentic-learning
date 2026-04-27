import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter

# 1. Prepare tiny dataset (replace with your own text)
text = """Hello, this is a simple RNN example for sequence modeling. We will train it on a small text dataset and generate new text based on what it learns. The model will learn to predict the next character in a sequence, allowing us to create new text that resembles the training data."""

# Character vocabulary
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")
print(f"Unique characters: {chars}")

# 2. Hyperparameters
seq_length = 15  # how many characters per sequence
hidden_size = 16
learning_rate = 1e-3
num_epochs = 1000  # we'll start small


# 3. Simple RNN class (from scratch)
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(vocab_size, hidden_size)  # input to hidden
        self.h2h = nn.Linear(hidden_size, hidden_size)  # hidden to hidden
        self.h2o = nn.Linear(hidden_size, vocab_size)  # hidden to output

    def forward(self, x, hidden):
        # x shape: (batch, vocab_size)  -- one-hot
        combined = self.i2h(x) + self.h2h(hidden)
        hidden = torch.tanh(combined)
        output = self.h2o(hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)


# 4. Training loop skeleton (you'll help complete this)
model = SimpleRNN(vocab_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 5. Data loader - creates (input_seq, target_seq) pairs
def create_data_loader(text, char_to_idx, seq_length, batch_size=32):
    """Create batches of (input_seq, target_seq) pairs"""
    encoded = [char_to_idx[ch] for ch in text]

    inputs = []
    targets = []

    for i in range(len(encoded) - seq_length):
        inputs.append(encoded[i : i + seq_length])
        targets.append(encoded[i + seq_length])

    num_samples = len(inputs)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for batch_start in range(0, num_samples, batch_size):
        batch_indices = indices[batch_start : batch_start + batch_size]

        batch_inputs = torch.zeros(len(batch_indices), seq_length, vocab_size)
        batch_targets = torch.zeros(len(batch_indices), dtype=torch.long)

        # One-hot encode inputs
        for j, idx in enumerate(batch_indices):
            for t, char_idx in enumerate(inputs[idx]):
                batch_inputs[j, t, char_idx] = 1.0
            batch_targets[j] = targets[idx]

        yield batch_inputs, batch_targets


# 6. Training loop with BPTT (Backpropagation Through Time)
print("\nTraining RNN...")
for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0

    for batch_inputs, batch_targets in create_data_loader(
        text, char_to_idx, seq_length
    ):
        # batch_inputs: (batch_size, seq_length, vocab_size)
        # batch_targets: (batch_size,)

        hidden = model.init_hidden(batch_size=batch_inputs.size(0))
        optimizer.zero_grad()

        # BPTT: process sequence step by step
        loss = 0
        for t in range(seq_length):
            output, hidden = model(batch_inputs[:, t, :], hidden)
            loss += criterion(output, batch_targets)

        # Backward pass through time
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


# 7. Generation function - sample new text
def generate_text(
    model, char_to_idx, idx_to_char, seed_text, length=100, temperature=0.8
):
    """Generate new text by sampling from the model"""
    model.eval()

    # Initialize with seed text - filter to only valid characters
    seed_text = seed_text.lower()
    seed_text = "".join([ch for ch in seed_text if ch in char_to_idx])
    if not seed_text:
        seed_text = list(char_to_idx.keys())[0]  # Use first char if seed is empty

    generated = seed_text
    hidden = model.init_hidden(batch_size=1)

    # Process seed text to set up hidden state
    for ch in seed_text:
        x = torch.zeros(1, vocab_size)
        x[0, char_to_idx[ch]] = 1.0
        with torch.no_grad():
            _, hidden = model(x, hidden)

    # Generate new characters
    for _ in range(length):
        # Last character of current text
        last_ch = generated[-1]
        x = torch.zeros(1, vocab_size)
        x[0, char_to_idx[last_ch]] = 1.0

        with torch.no_grad():
            output, hidden = model(x, hidden)

        # Apply temperature and sample
        logits = output[0] / temperature
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        next_idx = np.random.choice(vocab_size, p=probabilities)
        generated += idx_to_char[next_idx]

    return generated


# 8. Generate sample text
print("\n" + "=" * 50)
print("Generated Text Sample:")
print("=" * 50)
sample = generate_text(model, char_to_idx, idx_to_char, seed_text="Climate", length=200)
print(sample)
