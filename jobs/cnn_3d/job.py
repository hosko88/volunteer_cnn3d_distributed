import numpy as np

from framework.job import TrainingJob
from .model_numpy import AttentionCNN3D
from .datasets import CIFAR10Provider


class CNN3DJob(TrainingJob):
    """Job CNN 3D distribue (sans attention), reference / comparaison pour
    AttentionCNN3DJob. Meme pipeline de donnees reelles (CIFAR-10 pour
    la Phase 1), meme architecture Conv3D+FC, mais sans le repondage attentif
    des cartes de caracteristiques."""

    name = "CNN3D-distribue"

    def __init__(self, volume_shape=(8, 8, 8), n_classes=10, seed=0,
                 n_filters=4, kernel_size=3, data_provider=None):
        self.volume_shape = tuple(volume_shape)
        self.n_classes = n_classes
        self.seed = seed

        self.model = AttentionCNN3D(
            n_filters=n_filters, kernel_size=kernel_size, n_classes=n_classes,
            use_attention=False,
        )
        self.data = data_provider or CIFAR10Provider(
            volume_shape=self.volume_shape, n_classes=self.n_classes,
        )

    def kb_spec(self):
        return {"volume_shape": list(self.volume_shape), "n_classes": self.n_classes}

    @classmethod
    def from_kb_spec(cls, spec, cfg):
        return cls(volume_shape=tuple(spec.get("volume_shape", (8, 8, 8))),
                    n_classes=spec.get("n_classes", 10))

    def init_params(self):
        return self.model.init_params(seed=self.seed)

    def n_params(self):
        return self.model.n_params()

    def init_opt_state(self):
        return {"m": np.zeros(self.n_params(), dtype=np.float32),
                "v": np.zeros(self.n_params(), dtype=np.float32)}

    def make_task(self, epoch, index, seed, epsilon):
        return {"epoch": epoch, "index": index, "seed": int(seed),
                "batch_size": 4, "epsilon": float(epsilon)}

    def compute_gradient(self, params_flat, task):
        rng = np.random.default_rng(task["seed"])
        params = np.asarray(params_flat, dtype=np.float32)
        batch_size = int(task.get("batch_size", 4))

        x_batch, y_batch = self.data.sample_batch(rng, batch_size)

        grad_sum = np.zeros_like(params)
        loss_sum, correct = 0.0, 0
        for x, y in zip(x_batch, y_batch):
            g, loss, ok, _ = self.model.forward_backward(params, x, int(y))
            grad_sum += g
            loss_sum += loss
            correct += ok

        grad = (grad_sum / max(1, batch_size)).astype(np.float32)
        metrics = {
            "loss": float(loss_sum / max(1, batch_size)),
            "accuracy": float(correct / max(1, batch_size)),
        }
        return grad, batch_size, metrics

    def apply_gradient(self, params_flat, grad_flat, opt_state, lr,
                       weight_decay=1e-4, grad_clip=1.0):
        """AdamW + gradient clipping (aligne sur AttentionCNN3DJob)."""
        params = np.asarray(params_flat, dtype=np.float32)
        grad = np.asarray(grad_flat, dtype=np.float32)
        gnorm = float(np.linalg.norm(grad))
        if gnorm > grad_clip and gnorm > 0:
            grad = grad * (grad_clip / gnorm)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        opt_state["t"] = int(opt_state.get("t", 0)) + 1
        t = opt_state["t"]
        if "m" not in opt_state:
            opt_state["m"] = np.zeros_like(params)
            opt_state["v"] = np.zeros_like(params)
        opt_state["m"] = (beta1 * opt_state["m"] + (1 - beta1) * grad).astype(np.float32)
        opt_state["v"] = (beta2 * opt_state["v"] + (1 - beta2) * (grad * grad)).astype(np.float32)
        m_hat = opt_state["m"] / (1 - beta1 ** t)
        v_hat = opt_state["v"] / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        new_params = params - lr * update - lr * weight_decay * params
        return new_params.astype(np.float32), opt_state

    def evaluate(self, params_flat, n):
        params = np.asarray(params_flat, dtype=np.float32)
        x_eval, y_eval = self.data.sample_eval(n)
        if len(x_eval) == 0:
            return {"accuracy": 0.0, "top3": 0.0, "macro_f1": 0.0, "avg_turns": 1.0}

        correct, top3_correct = 0, 0
        for x, y in zip(x_eval, y_eval):
            probs = self.model.predict(params, x)
            correct += int(np.argmax(probs) == y)
            top3 = np.argsort(probs)[-3:]
            top3_correct += int(y in top3)

        acc = correct / len(x_eval)
        top3_acc = top3_correct / len(x_eval)
        return {"accuracy": float(acc), "top3": float(top3_acc),
                "macro_f1": float(acc), "avg_turns": 1.0}
