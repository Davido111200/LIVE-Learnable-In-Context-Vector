from .base_icv_encoder import BaseICVEncoder, ICVEncoderOutput
import torch

class GlobalICVEncoderOuter(BaseICVEncoder):
    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        alpha_learnable=True,
        alpha_init_value=0.0,
        use_sigmoid=False,
        load_from_coldstart=False,
        icv_cpk=None,
    ) -> None:
        """
        Initializes the GlobalICVEncoder to generate an ICV matrix via the outer product 
        of two learned vectors.

        There are two “networks”:
          - The first produces a tensor of shape (B, n_layers, 1).
          - The second produces a tensor of shape (B, 1, hidden_dim).

        Their outer product gives a matrix of shape (B, n_layers, hidden_dim). If each
        network’s outputs are initialized from N(0, 0.1^2), then the elementwise product 
        will have mean 0 and std ~ 0.01.

        Args:
            n_layers (int): Number of layers (the first dimension of the ICV matrix).
            hidden_dim (int): Hidden dimension (the second dimension of the ICV matrix).
            alpha_learnable (bool, optional): Whether alpha is learnable. Defaults to True.
            alpha_init_value (float, optional): Initial value of alpha. Defaults to 0.0.
            use_sigmoid (bool, optional): Whether to apply sigmoid activation to alpha. Defaults to False.
            load_from_coldstart (bool, optional): Whether to load from a cold start. Defaults to True.
        """
        super().__init__()

        if load_from_coldstart:
            self.vector1 = torch.nn.Parameter(icv_cpk["icv_encoder.vector1"])
            self.vector2 = torch.nn.Parameter(icv_cpk["icv_encoder.vector2"])
            self.alpha = torch.nn.Parameter(icv_cpk["icv_encoder.alpha"], requires_grad=alpha_learnable)
        else:
            # vector1: will be broadcast to (B, n_layers, 1)
            self.vector1 = torch.nn.Parameter(torch.empty(1, n_layers, 8))
            # vector2: will be broadcast to (B, 1, hidden_dim)
            self.vector2 = torch.nn.Parameter(torch.empty(1, 8, hidden_dim))

            # Alpha parameter (kept for potential use in other parts of the system).
            self.alpha = torch.nn.Parameter(
                torch.full(size=(1, n_layers), fill_value=float(alpha_init_value)),
                requires_grad=alpha_learnable,
            )

            # Initialize both so that each element ~ N(0, 0.1^2)
            torch.nn.init.normal_(self.vector1, mean=0.0, std=0.001)
            torch.nn.init.normal_(self.vector2, mean=0.0, std=0.001)

        self.use_sigmoid = use_sigmoid

    def forward(self, batch_size: int = 1) -> ICVEncoderOutput:
        """
        Produces the ICV matrix from the outer product of two vectors.
        
        Args:
            batch_size (int, optional): The batch size to use for expanding the learned parameters.
                Defaults to 1.
        
        Returns:
            ICVEncoderOutput: A dataclass (or similar) containing:
                - in_context_vector: The outer product of the two vectors with shape (B, n_layers, hidden_dim).
                - alpha: The alpha parameter (optionally passed through a sigmoid).
                - in_context_feature: None (reserved for future use).
        """
        # Expand each learned parameter to match the batch size.
        vec1 = self.vector1.expand(batch_size, -1, -1)  # Shape: (B, n_layers, 1)
        vec2 = self.vector2.expand(batch_size, -1, -1)   # Shape: (B, 1, hidden_dim)
        
        # Compute the outer product: broadcast multiplication results in shape (B, n_layers, hidden_dim)
        icv_matrix = vec1 @ vec2
        return ICVEncoderOutput(
            in_context_vector=icv_matrix,
            alpha=self.get_alpha(),
            in_context_feature=None
        )

    def get_alpha(self):
        return torch.sigmoid(self.alpha) if self.use_sigmoid else self.alpha
