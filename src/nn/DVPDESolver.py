import pennylane as qml
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from src.utils.logger import Logging
from src.nn.DVQuantumLayer import DVQuantumLayer
from src.nn.cheby_kan import ChebyPINN


class DVPDESolver(nn.Module):
    def __init__(self, args, logger: Logging, data=None, device=None):
        super().__init__()
        self.logger = logger
        self.device = device
        self.args = args
        self.data = data
        self.batch_size = self.args["batch_size"]
        self.num_qubits = self.args["num_qubits"]
        self.epochs = self.args["epochs"]
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        self.encoding = self.args.get("encoding", "angle")
        self.draw_quantum_circuit_flag = True
        self.classic_network = self.args["classic_network"]  # [3, 50, 50, 50, 4] #
        self.total_training_time = 0
        self.total_memory_peak = 0

        # Preprocessor type: "PIKAN" (Chebyshev-KAN) or "PINN" (plain MLP)
        self.network_type = self.args.get("network_type", "PIKAN").upper()

        # The preprocessor must output num_qubits features for the quantum embedding
        preprocessor_hidden = self.args.get("pikan_hidden", self.classic_network[-2]) \
            if self.network_type == "PIKAN" else self.classic_network[-2]
        preprocessor_network = [
            self.classic_network[0],
            preprocessor_hidden,
            self.num_qubits,
        ]
        self.preprocessor = self._build_preprocessor(preprocessor_network).to(
            self.device
        )

        self.postprocessor = nn.Sequential(
            nn.Linear(self.num_qubits, self.classic_network[-2]).to(self.device),
            nn.Tanh(),
            nn.Linear(self.classic_network[-2], self.classic_network[-1]).to(
                self.device
            ),
        ).to(self.device)

        #
        self.activation = nn.Tanh()

        self.num_qubits = args["num_qubits"]
        self.quantum_layer = DVQuantumLayer(self.args)

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.args["lr"]
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.9, patience=1000
        )

        self.loss_fn = torch.nn.MSELoss()

        self._initialize_logging()
        self._initialize_weights()

    def _build_preprocessor(self, network):
        """Build the classical preprocessor.

        Args:
            network (list[int]): Layer sizes, e.g. [in, hidden, num_qubits].

        Returns:
            nn.Module: a ChebyPINN (PIKAN) or an MLP (PINN) network.
        """
        if self.network_type == "PIKAN":
            self.logger.print("Using PIKAN (Chebyshev-KAN) preprocessor")
            return ChebyPINN(network)
        elif self.network_type == "PINN":
            self.logger.print("Using PINN (MLP) preprocessor")
            layers = []
            for i in range(len(network) - 1):
                layers.append(nn.Linear(network[i], network[i + 1]))
                if i < len(network) - 2:
                    layers.append(nn.Tanh())
            return nn.Sequential(*layers)
        else:
            raise ValueError(
                f"Unknown network_type '{self.network_type}'. "
                "Expected 'PIKAN' or 'PINN'."
            )

    def _initialize_weights(self):
        """ChebyPINN handles its own initialization; apply Xavier to MLP layers."""
        if self.network_type == "PINN":
            for layer in self.preprocessor:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _initialize_logging(self):
        self.log_path = self.logger.get_output_dir()
        # self.logger.print(f"checkpoint path: {self.log_path=}")

        # # total number of parameters
        # total_params = sum(p.numel() for p in self.parameters())
        # print(f"Total number of parameters: {total_params}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network
        Args:
            x: Spatial coordinates
            t: Time coordinates
        Returns:
            PDE solution values
        """

        try:
            if x.dim() != 2:
                raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
            # Combine inputs
            # Classical preprocessing
            preprocessed = self.preprocessor(x)
            # print(f"preprocessed: {preprocessed.shape}")
            # Quantum processing

            if self.draw_quantum_circuit_flag:
                self.draw_quantum_circuit(preprocessed)
                self.draw_quantum_circuit_flag = False

            quantum_out = self.quantum_layer(preprocessed).to(
                dtype=torch.float32, device=self.device
            )
            # print(f"quantum_out: {quantum_out.shape}")### ahapw is 1, batch_size*n_qubits
            quantum_features = quantum_out.view( self.num_qubits , -1).T
            # print(f"quantum_out: {quantum_features.shape}")### ahapw is 1, batch_size*n_qubits

            classical_out = self.postprocessor(quantum_features)
            # print(f"classical_out: {classical_out.shape}")
            return classical_out

        except Exception as e:
            self.logger.print(f"Forward pass failed: {str(e)}")
            raise

    def save_state(self , path=None):
        state = {
            "args": self.args,
            "classic_network": self.classic_network,
            "quantum_params": self.quantum_layer.state_dict(),
            "preprocessor": self.preprocessor.state_dict(),
            "quantum_layer": self.quantum_layer.state_dict(),
            "postprocessor": self.postprocessor.state_dict(),
            # "classical_input_scale": self.classical_input_scale.detach().cpu().numpy(),
            # "classical_output_scale": self.classical_output_scale.detach().cpu().numpy(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "log_path": self.log_path,
        }

        if path is None:
            model_path = os.path.join(self.log_path, "model.pth")
        else:    
            model_path = path

        with open(model_path, "wb") as f:
            torch.save(state, f)

        self.logger.print(f"Model state saved to {model_path}")

    @classmethod
    def load_state(cls, file_path, map_location=None):
        if map_location is None:
            map_location = torch.device("cpu")
        with open(file_path, "rb") as f:
            state = torch.load(f, map_location=map_location)
            # state = pickle.load(f)
        # print(f"Model state loaded from {file_path}")
        return state

    def draw_quantum_circuit(self, x):
        if self.draw_quantum_circuit_flag:
            try:
                self.logger.print("The circuit used in the study:")
                if self.quantum_layer.params is not None:
                    fig, ax = qml.draw_mpl(self.quantum_layer.circuit)(x[0])
                    plt.savefig(os.path.join(self.log_path, "circuit.pdf"))
                    plt.close()
                    print(f"The circuit is saved in {self.log_path}")
            except Exception as e:
                self.logger.print(f"Failed to draw quantum circuit: {str(e)}")
