# File name: ketos_frozen_bn.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""``ketos`` with every BatchNorm layer frozen in evaluation mode (running statistics untouched).

Why: fine-tuning the PP-OCRv6 base with ``ketos train`` collapsed its validation accuracy from 90 % to 9 %
even at a learning rate of 1e-8.  A tensor-by-tensor comparison of the base and the "trained" weights showed
every weight unchanged to within 0.3 % while 103 of the 122 BatchNorm running statistics had moved by up to
76 %: kraken's trainer puts the whole network into train mode, so the running mean/variance are re-estimated
from the first ~100 small batches of the new data and the pretrained features no longer see the normalisation
they were trained with.  Freezing BatchNorm (eval mode in the forward pass, affine weights still trainable) is
the usual remedy for small-batch fine-tuning.  ``ketos`` has no flag for it, so this wrapper patches
:meth:`torch.nn.modules.batchnorm._BatchNorm.train` to ignore ``mode=True`` and then runs the ordinary ketos
command line unchanged::

    python -m src.finetuning.kraken.ketos_frozen_bn -d mps --threads 4 --workers 0 train -f binary -i base.safetensors ...

Every ketos option works as usual; only the BatchNorm behaviour differs.  Set ``KETOS_FROZEN_BN=0`` to run
the unpatched ketos through the same entry point (for A/B runs with identical command lines).
"""
import os
import sys
from typing import Any, List, Optional


def freeze_batchnorm() -> None:
    """Pin every ``_BatchNorm`` to inference behaviour: ``train()`` keeps it in eval mode and, as a belt
    and braces, ``forward`` always normalises with the stored running statistics (``training=False``), so
    the running mean/variance and ``num_batches_tracked`` can never change, whatever a trainer sets on
    the ``training`` flag.  The affine weight and bias still receive gradients.

    :return: None.
    :rtype: None
    """
    import torch.nn.functional as F
    import torch.nn.modules.batchnorm as batchnorm

    original_train = batchnorm._BatchNorm.train

    def frozen_train(self: Any, mode: bool = True) -> Any:
        """``train()`` replacement: BatchNorm always ends up in eval mode.

        :param self: The BatchNorm module.
        :param mode: Ignored (the module is always put in eval mode).
        :type mode: bool
        :return: The module.
        """
        return original_train(self, False)

    def frozen_forward(self: Any, input: Any) -> Any:
        """``forward`` replacement: normalise with the running statistics, never update them.

        :param self: The BatchNorm module.
        :param input: Input tensor.
        :return: Normalised tensor.
        """
        self._check_input_dim(input)
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)

    batchnorm._BatchNorm.train = frozen_train  # type: ignore[assignment]
    batchnorm._BatchNorm.forward = frozen_forward  # type: ignore[assignment]


def main(argv: Optional[List[str]] = None) -> None:
    """Run the ketos command line, with BatchNorm frozen unless ``KETOS_FROZEN_BN=0``.

    :param argv: ketos arguments (default: ``sys.argv[1:]``).
    :type argv: Optional[List[str]]
    :return: None.
    :rtype: None
    """
    if os.environ.get("KETOS_FROZEN_BN", "1") != "0":
        freeze_batchnorm()
        print("[ketos_frozen_bn] BatchNorm layers frozen (eval mode + running statistics pinned)", file=sys.stderr, flush=True)
    from kraken.ketos import cli

    sys.argv = ["ketos"] + list(sys.argv[1:] if argv is None else argv)
    cli()


if __name__ == "__main__":
    main()
