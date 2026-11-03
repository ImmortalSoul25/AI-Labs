# Laboratory — Neural Models
## Learning, Depth, Activations, and Output Layers

**Implementation:** PyTorch, CPU  
**Reproducibility:** random seed 0, full-batch training, Adam, learning rate 0.03, 5,000 updates.

## 1. Task 1 — XOR problem and separability

The binary dataset is:

| Input | XOR label |
|---|---:|
| (0, 0) | 0 |
| (0, 1) | 1 |
| (1, 0) | 1 |
| (1, 1) | 0 |

The two positive examples lie on opposite corners of the unit square, while the negative examples occupy the other two corners. No single straight line can separate the positive pair from the negative pair. Therefore, an affine model followed by sigmoid cannot represent XOR exactly: sigmoid changes the output scale but does not make the decision boundary nonlinear.

A hidden layer with a nonlinear activation can transform the inputs into a representation that makes the classes separable.

## 2. Task 2 — Binary network design

The implementation uses the required **2–2–1** architecture:
- Input: two features.
- Hidden layer: two units with **tanh** activation.
- Output: one raw logit.
- Loss: `BCEWithLogitsLoss`, which combines sigmoid and binary cross-entropy in a numerically stable operation.
- Optimizer: Adam, learning rate 0.03; full-batch training for 5,000 steps.

The model returns a logit rather than applying sigmoid inside `forward`. Sigmoid is applied only when probabilities are needed for display or thresholding.

### Validation checks

1. Final binary loss is much lower than initial loss.
2. All four predicted labels match the XOR targets.
3. The first-layer weights receive nonzero gradients during the first backward pass.

## 3. Task 3 — LLM prompt and verification

**Prompt used to prepare the implementation:**

> Write a concise, runnable PyTorch implementation for the XOR neural-model laboratory. Use exactly a 2-input, 2-hidden-unit, 1-output network. Use XOR inputs and binary labels, a nonlinear hidden activation, a single output logit with BCEWithLogitsLoss, random parameter initialization, full-batch training for a few thousand CPU steps, and print initial/final loss, the four predicted probabilities and labels, and a first-layer gradient tensor. Also include a zero-initialization symmetry experiment, compare sigmoid/tanh/ReLU hidden activations, and implement the specified three-class mapping using three output logits and CrossEntropyLoss. Keep the code readable and explain the output/loss pairing.

The generated implementation was checked by running it. The architecture dimensions, target encodings, loss interfaces, output shapes, probability sums, and predictions were inspected. The reported measurements below are from that run, not estimated values.

## 4. Task 4 — Binary results and gradients

### Main XOR run (tanh)

| Measurement | Result |
|---|---:|
| Initial loss | 0.715159 |
| Final loss | 0.000048 |
| Accuracy | 4/4 (100%) |

Predicted probabilities in the input order (00, 01, 10, 11):

`[0.0000342, 0.9999411, 0.9999323, 0.0000294]`

Predicted labels: `[0, 1, 1, 0]`, matching the target labels.

The first-layer weight gradient on the first backward pass was:

```text
[[ 0.00050540,  0.00060695],
 [-0.04256904, -0.04477292]]
```

Its Frobenius norm was **0.061785**. This nonzero gradient is evidence that the loss is being propagated through the output layer and hidden activation into the first layer. Backpropagation computes how a small change in each parameter affects the loss; the optimizer then uses these gradients to update the parameters.

### Zero-initialization symmetry experiment

All model parameters were set to zero before the backward pass. The two hidden units received identical weight gradients and identical bias gradients (all zero in this particular setup).

When hidden units start with identical parameters, they produce identical activations. Because their downstream connections are also identical, they receive the same gradient and remain identical after an update. They therefore cannot learn distinct features. Random initialization breaks this symmetry by giving hidden units different starting parameters.

### Activation comparison

All three models used the same seed, architecture, optimizer, learning rate, and 5,000 training steps.

| Hidden activation | Initial loss | Final loss | Accuracy | First-layer gradient norm (first step) |
|---|---:|---:|---:|---:|
| Sigmoid | 0.697659 | 0.000145 | 4/4 | 0.000891 |
| Tanh | 0.715159 | 0.000048 | 4/4 | 0.061785 |
| ReLU | 0.706885 | 0.477388 | 3/4 | 0.001699 |

For this seed and training setup, sigmoid and tanh both learned XOR, while ReLU did not reach perfect accuracy. Tanh had the largest first-layer gradient norm in this run. This is one experiment, not proof that one activation is always best: activation performance depends on initialization, data, architecture, and optimization settings. Saturated sigmoid/tanh units can have small derivatives; ReLU has zero derivative for negative pre-activations.

## 5. Task 5 — Three-class output layer

The class mapping is:
- Class 0: (0, 0)
- Class 1: disagreement — (0, 1) and (1, 0)
- Class 2: (1, 1)

The network retains two hidden units and produces **three logits**. The final linear layer has weight shape **(3, 2)** and bias shape **(3,)**. `CrossEntropyLoss` receives raw logits and integer class labels; it internally performs the appropriate log-softmax/cross-entropy calculation. A separate softmax is used only to display probabilities.

| Measurement | Result |
|---|---:|
| Initial loss | 1.059108 |
| Final loss | 0.000024 |
| Accuracy | 4/4 (100%) |

Softmax probability rows summed to approximately 1 (floating-point values between 0.99999994 and 1.0). Adding 100 to every logit in a row left its softmax probabilities unchanged, as expected because softmax is invariant to a shared additive shift.

For a softmax-cross-entropy classification example, the gradient with respect to each logit is (p-y), where (p) is the predicted probability vector and (y) is the one-hot target vector (with the batch reduction applied by the loss).

## 6. Reflection

**Why does depth help, and why is nonlinearity important?**  
A stack of affine layers without nonlinear activations is still just one affine transformation. A nonlinear hidden layer can create intermediate features and a nonlinear decision boundary, which is needed for XOR.

**What is evidence that backpropagation is working?**  
The first-layer gradient is nonzero, and optimization reduces the binary loss from 0.715159 to 0.000048 while producing all four correct labels.

**Why does identical initialization cause a problem?**  
Identical hidden units compute the same function and receive the same updates. They do not diversify into different features. Random initialization breaks this symmetry.

**How did activation choice affect learning?**  
In this run tanh and sigmoid reached 4/4 correct, while ReLU reached 3/4. Their initial gradient norms also differed. This illustrates why gradient flow and activation choice should be checked experimentally rather than assumed.

**Why must output activation and loss be selected together?**  
Binary classification uses one logit with `BCEWithLogitsLoss`; multiclass classification uses a vector of logits with `CrossEntropyLoss`. These losses already include the corresponding sigmoid or softmax-related calculation internally, so passing probabilities instead of logits would be the wrong interface.

**How should LLM-generated code be used?**  
An LLM can accelerate boilerplate and suggest experiments, but its output must be verified: check tensor shapes, labels, loss expectations, gradients, numerical results, and whether the code actually runs. The reported numbers here were obtained by executing the script.

**How can tests scale without exhaustive finite differences?**  
Use small deterministic sanity checks (known labels, tensor shapes, finite losses, probability sums), gradient inspection, and selected numerical gradient checks on a few parameters. Full finite-difference checking of every parameter becomes expensive for larger models.

## 7. Files and execution

- `neural_models_lab.py` — complete runnable implementation.
- `REPORT.md` — this report.

Install dependencies with `pip install torch numpy` and run `python neural_models_lab.py`. The code runs on CPU and does not require a GPU.
