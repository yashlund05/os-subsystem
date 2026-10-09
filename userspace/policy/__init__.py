"""User-space policy table package.

.. deprecated::
    DEPRECATED MODULE (remediation item 1.4) — 2026-10-08

    ``userspace/policy/`` is an unreferenced reimplementation that is NOT called
    from any live code path or test.  The active policy representation is used
    directly within the kernel fast path via ``kernel/include/neuroos_kernel.h``
    (``struct neuroos_quantized_policy``).

    Do NOT edit this module expecting changes to take effect in dispatch.

    This module will be deleted in a future cleanup pass.
"""

import warnings

from userspace.policy.policy_table import DoubleBufferedPolicyTable, PolicySnapshot

warnings.warn(
    "userspace.policy is DEPRECATED and unreferenced. "
    "The live policy struct is defined in kernel/include/neuroos_kernel.h. "
    "(Remediation item 1.4 — will be removed in a future cleanup.)",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DoubleBufferedPolicyTable", "PolicySnapshot"]
