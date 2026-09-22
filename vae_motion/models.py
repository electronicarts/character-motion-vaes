import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.controller import init, DiagGaussian
from common.model_io import (
    CheckpointFormatError,
    load_tensors_with_config,
    read_config_from_file,
    save_tensors_with_config,
)


class NormalizationMixin:
    """Shared normalization state for the VAE models.

    The statistics are registered as buffers so they appear in state_dict and
    follow .to(device). They used to be plain attributes, which meant a
    state_dict-based save silently dropped them.
    """

    STAT_KEYS = ("max", "min", "avg", "std")

    def _init_normalization(self, normalization):
        self.mode = normalization.get("mode")
        for key in self.STAT_KEYS:
            value = normalization.get(key)
            if value is not None:
                value = torch.as_tensor(value).float()
            self.register_buffer("data_" + key, value)

    @staticmethod
    def normalization_from_tensors(config, tensors):
        """Rebuild the `normalization` constructor argument from a checkpoint."""
        normalization = {"mode": config["normalization_mode"]}
        for key in NormalizationMixin.STAT_KEYS:
            name = "data_" + key
            if name in tensors:
                normalization[key] = tensors[name]
        return normalization

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")


class AutoEncoder(NormalizationMixin, nn.Module):
    def __init__(self, frame_size, latent_size, normalization):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size

        self._init_normalization(normalization)

        h1 = 256
        h2 = 128
        # Encoder
        # Takes pose | condition (n * poses) as input
        self.fc1 = nn.Linear(frame_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, latent_size)

        # Decoder
        # Takes latent | condition as input
        self.fc4 = nn.Linear(latent_size, h2)
        self.fc5 = nn.Linear(h2, h1)
        self.fc6 = nn.Linear(h1, frame_size)

    @property
    def config(self):
        return {
            "frame_size": self.frame_size,
            "latent_size": self.latent_size,
            "normalization_mode": self.mode,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(
            config["frame_size"],
            config["latent_size"],
            cls.normalization_from_tensors(config, tensors),
        )

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

    def decode(self, x):
        h4 = F.relu(self.fc4(x))
        h5 = F.relu(self.fc5(h4))
        return self.fc6(h5)


class Encoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        input_size = frame_size * (num_future_predictions + num_condition_frames)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(frame_size + hidden_size, hidden_size)
        self.mu = nn.Linear(frame_size + hidden_size, latent_size)
        self.logvar = nn.Linear(frame_size + hidden_size, latent_size)

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Decoder
        # Takes latent | condition as input
        input_size = latent_size + frame_size * num_condition_frames
        output_size = num_future_predictions * frame_size
        self.fc4 = nn.Linear(input_size, hidden_size)
        self.fc5 = nn.Linear(latent_size + hidden_size, hidden_size)
        self.out = nn.Linear(latent_size + hidden_size, output_size)

    def decode(self, z, c):
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        return self.out(torch.cat((z, h5), dim=1))

    def forward(self, z, c):
        return self.decode(z, c)


class MixedDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + frame_size * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = num_future_predictions * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        layer_out = c

        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out


class PoseMixtureVAE(NormalizationMixin, nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self._init_normalization(normalization)

        hidden_size = 256
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = Encoder(*args)
        self.decoder = MixedDecoder(*args, num_experts)

    @property
    def config(self):
        return {
            "frame_size": self.frame_size,
            "latent_size": self.latent_size,
            "num_condition_frames": self.num_condition_frames,
            "num_future_predictions": self.num_future_predictions,
            # num_experts is not kept as an attribute; the first expert weight
            # has shape (num_experts, input_size, hidden_size).
            "num_experts": int(self.decoder.w0.shape[0]),
            "normalization_mode": self.mode,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(
            config["frame_size"],
            config["latent_size"],
            config["num_condition_frames"],
            config["num_future_predictions"],
            cls.normalization_from_tensors(config, tensors),
            config["num_experts"],
        )

    def encode(self, x, c):
        _, mu, logvar = self.encoder(x, c)
        return mu, logvar

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return self.decoder(z, c), mu, logvar

    def sample(self, z, c, deterministic=False):
        return self.decoder(z, c)


class PoseMixtureSpecialistVAE(NormalizationMixin, nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self._init_normalization(normalization)

        hidden_size = 128
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = Encoder(*args)

        self.decoders = []
        for i in range(num_experts):
            decoder = Decoder(*args)
            self.decoders.append(decoder)
            self.add_module("d" + str(i), decoder)

        # Gating network
        gate_hsize = 128
        input_size = latent_size + frame_size * num_condition_frames
        self.g_fc1 = nn.Linear(input_size, gate_hsize)
        self.g_fc2 = nn.Linear(latent_size + gate_hsize, gate_hsize)
        self.g_fc3 = nn.Linear(latent_size + gate_hsize, num_experts)

    @property
    def config(self):
        return {
            "frame_size": self.frame_size,
            "latent_size": self.latent_size,
            "num_condition_frames": self.num_condition_frames,
            "num_future_predictions": self.num_future_predictions,
            "num_experts": len(self.decoders),
            "normalization_mode": self.mode,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(
            config["frame_size"],
            config["latent_size"],
            config["num_condition_frames"],
            config["num_future_predictions"],
            cls.normalization_from_tensors(config, tensors),
            config["num_experts"],
        )

    def gate(self, z, c):
        h1 = F.elu(self.g_fc1(torch.cat((z, c), dim=1)))
        h2 = F.elu(self.g_fc2(torch.cat((z, h1), dim=1)))
        return self.g_fc3(torch.cat((z, h2), dim=1))

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        coefficients = F.softmax(self.gate(z, c), dim=1)
        predictions = torch.stack([decoder(z, c) for decoder in self.decoders], dim=1)
        return predictions, mu, logvar, coefficients

    def sample(self, z, c, deterministic=False):
        coefficients = F.softmax(self.gate(z, c), dim=1)
        predictions = torch.stack([decoder(z, c) for decoder in self.decoders], dim=1)

        if not deterministic:
            dist = torch.distributions.Categorical(coefficients)
            indices = dist.sample()
        else:
            indices = coefficients.argmax(dim=1)

        return predictions[torch.arange(predictions.size(0)), indices]


class PoseVAE(NormalizationMixin, nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self._init_normalization(normalization)

        h1 = 256
        # Encoder
        # Takes pose | condition (n * poses) as input
        self.fc1 = nn.Linear(
            frame_size * (num_future_predictions + num_condition_frames), h1
        )
        self.fc2 = nn.Linear(frame_size + h1, h1)
        # self.fc3 = nn.Linear(h1, h1)
        self.mu = nn.Linear(frame_size + h1, latent_size)
        self.logvar = nn.Linear(frame_size + h1, latent_size)

        # Decoder
        # Takes latent | condition as input
        self.fc4 = nn.Linear(latent_size + frame_size * num_condition_frames, h1)
        self.fc5 = nn.Linear(latent_size + h1, h1)
        # self.fc6 = nn.Linear(latent_size + h1, h1)
        self.out = nn.Linear(latent_size + h1, num_future_predictions * frame_size)

    @property
    def config(self):
        return {
            "frame_size": self.frame_size,
            "latent_size": self.latent_size,
            "num_condition_frames": self.num_condition_frames,
            "num_future_predictions": self.num_future_predictions,
            "normalization_mode": self.mode,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(
            config["frame_size"],
            config["latent_size"],
            config["num_condition_frames"],
            config["num_future_predictions"],
            cls.normalization_from_tensors(config, tensors),
        )

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        # h3 = F.elu(self.fc3(h2))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def decode(self, z, c):
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        # h6 = F.elu(self.fc6(torch.cat((z, h5), dim=1)))
        return self.out(torch.cat((z, h5), dim=1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample(self, z, c, deterministic=False):
        return self.decode(z, c)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, latent_size):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.latent_size = latent_size

        # self.embedding = nn.Embedding(self.num_embeddings, self.latent_size)
        # self.embedding.weight.data.normal_()

        embed = torch.randn(latent_size, num_embeddings)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

        self.commitment_cost = 0.25
        self.decay = 0.99
        self.epsilon = 1e-5

    def forward(self, inputs):
        # Calculate distances
        dist = (
            inputs.pow(2).sum(1, keepdim=True)
            - 2 * inputs @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.num_embeddings).type(inputs.dtype)
        embed_ind = embed_ind.view(*inputs.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        # Use EMA to update the embedding vectors
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )

            embed_sum = inputs.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        loss = (quantize.detach() - inputs).pow(2).mean()
        quantize = inputs + (quantize - inputs).detach()

        avg_probs = embed_onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * (avg_probs + 1e-10).log()))

        return quantize, loss, perplexity, embed_ind


class PoseVQVAE(NormalizationMixin, nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_embeddings,
        num_condition_frames,
        num_future_predictions,
        normalization,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self._init_normalization(normalization)

        h1 = 512
        # Encoder
        # Takes pose | condition (n * poses) as input
        self.fc1 = nn.Linear(
            frame_size * (num_future_predictions + num_condition_frames), h1
        )
        self.fc2 = nn.Linear(h1, h1)
        self.fc3 = nn.Linear(h1, h1)
        self.mu = nn.Linear(h1, latent_size)

        # Decoder
        # Takes latent | condition as input
        self.fc4 = nn.Linear(latent_size + frame_size * num_condition_frames, h1)
        self.fc5 = nn.Linear(h1, h1)
        self.fc6 = nn.Linear(h1, h1)
        self.out = nn.Linear(h1, num_future_predictions * frame_size)

        self.quantizer = VectorQuantizer(num_embeddings, latent_size)

    @property
    def config(self):
        return {
            "frame_size": self.frame_size,
            "latent_size": self.latent_size,
            "num_embeddings": self.quantizer.num_embeddings,
            "num_condition_frames": self.num_condition_frames,
            "num_future_predictions": self.num_future_predictions,
            "normalization_mode": self.mode,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(
            config["frame_size"],
            config["latent_size"],
            config["num_embeddings"],
            config["num_condition_frames"],
            config["num_future_predictions"],
            cls.normalization_from_tensors(config, tensors),
        )

    def forward(self, x, c):
        mu = self.encode(x, c)
        quantized, loss, perplexity, _ = self.quantizer(mu)
        recon = self.decode(quantized, c)
        return recon, loss, perplexity

    def encode(self, x, c):
        s = torch.cat((x, c), dim=1)
        h1 = F.relu(self.fc1(s))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return self.mu(h3)

    def decode(self, z, c):
        s = torch.cat((z, c), dim=1)
        h4 = F.relu(self.fc4(s))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        return self.out(h6)

    def sample(self, z, c, deterministic=False):
        if not deterministic:
            dist = torch.distributions.Categorical(z.softmax(dim=1))
            indices = dist.sample()
        else:
            indices = z.argmax(dim=1)
        z = F.embedding(indices, self.quantizer.embed.transpose(0, 1))
        s = torch.cat((z, c), dim=1)
        h4 = F.relu(self.fc4(s))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        return self.out(h6)


class PoseVAEController(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        init_r_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        init_s_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("sigmoid"),
        )
        init_t_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("tanh"),
        )

        h_size = 256
        self.actor = nn.Sequential(
            init_r_(nn.Linear(self.observation_dim, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_t_(nn.Linear(h_size, self.action_dim)),
            nn.Tanh(),
        )

    @property
    def config(self):
        return {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(config["observation_dim"], config["action_dim"])

    def forward(self, x):
        return self.actor(x)


class PoseVAEPolicy(nn.Module):
    def __init__(self, controller):
        super().__init__()
        self.actor = controller
        self.dist = DiagGaussian(controller.action_dim, controller.action_dim)

        init_s_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("sigmoid"),
        )
        init_r_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        h_size = 256
        self.critic = nn.Sequential(
            init_r_(nn.Linear(controller.observation_dim, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_s_(nn.Linear(h_size, 1)),
        )
        self.state_size = 1

    @property
    def config(self):
        return {
            "observation_dim": self.actor.observation_dim,
            "action_dim": self.actor.action_dim,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(PoseVAEController(config["observation_dim"], config["action_dim"]))

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        action = self.actor(inputs)
        dist = self.dist(action)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
            action.clamp_(-1.0, 1.0)

        action_log_probs = dist.log_probs(action)
        value = self.critic(inputs)

        return value, action, action_log_probs

    def get_value(self, inputs):
        value = self.critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value = self.critic(inputs)
        mode = self.actor(inputs)
        dist = self.dist(mode)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


MODEL_REGISTRY = {
    "AutoEncoder": AutoEncoder,
    "PoseVAE": PoseVAE,
    "PoseMixtureVAE": PoseMixtureVAE,
    "PoseMixtureSpecialistVAE": PoseMixtureSpecialistVAE,
    "PoseVQVAE": PoseVQVAE,
    "PoseVAEController": PoseVAEController,
    "PoseVAEPolicy": PoseVAEPolicy,
}


def save_model(model, path, extra_config=None):
    """Write `model` to `path` as a safetensors checkpoint."""
    config = dict(model.config)
    config["class"] = type(model).__name__
    if extra_config:
        config.update(extra_config)
    save_tensors_with_config(model.state_dict(), config, path)


def load_model(path, device="cpu"):
    """Rebuild the model described by the checkpoint at `path`."""
    tensors, config = load_tensors_with_config(path, device)
    class_name = config.get("class")
    model_class = MODEL_REGISTRY.get(class_name)
    if model_class is None:
        raise CheckpointFormatError(
            "{} declares unknown model class {!r}; known classes are {}".format(
                path, class_name, sorted(MODEL_REGISTRY)
            )
        )
    # Building from config first means the normalization buffers already exist
    # at the right shape, so the strict load below is a no-op for them.
    model = model_class.from_config(config, tensors)
    model.load_state_dict(tensors, strict=True)
    return model.to(device)


def read_config(path):
    """Return a checkpoint's config without reading its tensors."""
    return read_config_from_file(path)
