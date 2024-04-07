import os
import sys

from options.base_option import BaseOptions

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainOptions(BaseOptions):
    """Train options"""

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        """Initialize train options"""
        BaseOptions.initialize(self)

        self._parser.add_argument(
            "--model_name",
            type=str,
            required=False,
            default="BlurGAN",
            help="model name",
            choices=[
                "ACGAN",
                "VanillaGAN",
                "ACVanillaGAN",
                "WGANGP",
                "WGANWC",
                "MorphGAN",
                "WCGANGP",
                "BlurGAN",
                "STYLEGAN",
            ],
        )
        self._parser.add_argument(
            "--init_type",
            type=str,
            required=False,
            default="xavier_normal",
            help="initialization type",
            choices=["normal", "xavier_normal", "kaiming_normal"],
        )

        self._parser.add_argument(
            "--optimizer",
            type=str,
            required=False,
            default="rmsprop",
            help="optimizer",
            choices=["adam", "rmsprop"],
        )
        self._parser.add_argument(
            "--lr_scheduler",
            type=str,
            required=False,
            default="none",
            help="learning rate scheduler",
            choices=["none", "step"],
        )
        self._parser.add_argument(
            "--lr_decay_start",
            type=int,
            required=False,
            default=10,
            help="learning rate decay start epoch",
        )
        self._parser.add_argument(
            "--lr_decay_step",
            type=int,
            required=False,
            default=2,
            help="learning rate decay step",
        )
        self._parser.add_argument(
            "--lr_decay_gamma",
            type=float,
            required=False,
            default=0.3,
            help="learning rate decay gamma",
        )
        self._parser.add_argument(
            "--g_lr",
            type=float,
            required=False,
            default=0.0003,
            help="generator learning rate",
        )
        self._parser.add_argument(
            "--g_adam_beta1",
            type=float,
            required=False,
            default=0.5,
            help="generator adam beta1",
        )
        self._parser.add_argument(
            "--g_adam_beta2",
            type=float,
            required=False,
            default=0.999,
            help="generator adam beta2",
        )

        self._parser.add_argument(
            "--d_lr",
            type=float,
            required=False,
            default=0.0003,
            help="discriminator learning rate",
        )
        self._parser.add_argument(
            "--d_adam_beta1",
            type=float,
            required=False,
            default=0.5,
            help="discriminator adam beta1",
        )
        self._parser.add_argument(
            "--d_adam_beta2",
            type=float,
            required=False,
            default=0.999,
            help="discriminator adam beta2",
        )
        self._parser.add_argument(
            "--latent_dim",
            type=int,
            required=False,
            default=112,
            help="latent dimension",
        )
        self._parser.add_argument(
            "--embedding_dim",
            type=int,
            required=False,
            default=112,
            help="embedding dimension",
        )
        self._parser.add_argument(
            "--n_classes",
            type=int,
            required=False,
            default=5,
            help="number of classes for conditional GANs",
        )

        # VanillaGAN and ACVanillaGAN parameters
        self._parser.add_argument(
            "--vanilla_g_neurons",
            type=int,
            required=False,
            default=256,
            help="VanillaGAN generator neurons",
        )
        self._parser.add_argument(
            "--vanilla_d_neurons",
            type=int,
            required=False,
            default=1024,
            help="VanillaGAN discriminator neurons",
        )
        self._parser.add_argument(
            "--d_lambda_adv",
            type=float,
            required=False,
            default=1,
            help="discriminator adversarial loss weight",
        )
        self._parser.add_argument(
            "--d_lambda_cls",
            type=float,
            required=False,
            default=1,
            help="discriminator classification loss weight",
        )
        self._parser.add_argument(
            "--g_lambda_adv",
            type=float,
            required=False,
            default=1,
            help="generator adversarial loss weight",
        )
        self._parser.add_argument(
            "--g_lambda_cls",
            type=float,
            required=False,
            default=1,
            help="generator classification loss weight",
        )

        # WGAN parameters
        self._parser.add_argument(
            "--d_lambda_w",
            type=float,
            required=False,
            default=1,
            help="discriminator wasserstein loss weight",
        )
        self._parser.add_argument(
            "--g_lambda_w",
            type=float,
            required=False,
            default=1,
            help="generator wasserstein loss weight",
        )
        self._parser.add_argument(
            "--d_lambda_gp",
            type=float,
            required=False,
            default=10,
            help="discriminator gradient penalty loss weight",
        )
        self._parser.add_argument(
            "--clip_value",
            type=float,
            required=False,
            default=0.01,
            help="weight clipping value",
        )

        

        ######## New Parameters to be added above this line ########
        self._is_train = True
