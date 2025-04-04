from .base_icv_encoder import BaseICVEncoder, ICVEncoderOutput
import torch


class GlobalICVEncoder(BaseICVEncoder):
    def __init__(
        self,
        lmm_hidden_dim,
        lmm_layers,
        alpha_learnable=True,
        alpha_init_value=0.0,
        use_sigmoid=False,
        icv_cpk=None,
        load_from_coldstart=False,
    ) -> None:
        """
        Initializes the GlobalICVEncoder object.
        
        Args:
            lmm_hidden_dim (int): The hidden dimension of the LMM layers.
            lmm_layers (int): The number of LMM layers.
            alpha_learnable (bool, optional): Whether the alpha parameter is learnable. Defaults to True.
            alpha_init_value (float, optional): The initial value of the alpha parameter. Defaults to 0.0.
            use_sigmoid (bool, optional): Whether to use sigmoid activation. Defaults to False.
        """
        super().__init__()

        if load_from_coldstart:
            self.alpha = torch.nn.Parameter(icv_cpk["alpha"], requires_grad=alpha_learnable)
            self.vector1 = torch.nn.Parameter(icv_cpk["vector1"], requires_grad=True)
            self.vector2 = torch.nn.Parameter(icv_cpk["vector2"], requires_grad=True)
        else:
            self.alpha = torch.nn.Parameter(
                torch.full(size=(1, lmm_layers), fill_value=float(alpha_init_value)),
                requires_grad=alpha_learnable,
            )
            self.icv = torch.nn.Parameter(torch.empty(1, lmm_layers, lmm_hidden_dim))
            torch.nn.init.normal_(self.icv, mean=0.0, std=0.01)

            # # vector1: will be broadcast to (B, n_layers, 1)
            # self.vector1 = torch.nn.Parameter(torch.empty(1, lmm_layers, 8), requires_grad=True)
            # # vector2: will be broadcast to (B, 1, hidden_dim)
            # self.vector2 = torch.nn.Parameter(torch.empty(1, 8, lmm_hidden_dim), requires_grad=True)

            # torch.nn.init.normal_(self.vector1, mean=0.0, std=0.01)
            # torch.nn.init.normal_(self.vector2, mean=0.0, std=0.01)

        self.use_sigmoid = use_sigmoid
        

    def forward(self) -> ICVEncoderOutput:
        # Expand each learned parameter to match the batch size.
        # vec1 = self.vector1.expand(1, -1, -1)  # Shape: (1, n_layers, 1)
        # vec2 = self.vector2.expand(1, -1, -1)   # Shape: (1, 1, hidden_dim)
        
        # # Compute the outer product: broadcast multiplication results in shape (B, n_layers, hidden_dim)
        # icv_matrix = vec1 @ vec2
        # return ICVEncoderOutput(
        #     in_context_vector=icv_matrix, alpha=self.get_alpha(), in_context_feature=None
        # )
        return ICVEncoderOutput(
            in_context_vector=self.icv, alpha=self.get_alpha(), in_context_feature=None
        )

    def get_alpha(self):
        if self.use_sigmoid:
            return torch.sigmoid(self.alpha)
        return self.alpha
