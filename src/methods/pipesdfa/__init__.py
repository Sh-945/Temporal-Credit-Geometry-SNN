"""PipeSDFA software reference.

The pipeline changes scheduling/hardware overlap, not the dense feedback
equation; the training engine therefore uses the same numerical local rule as
sDFA and records ``pipesdfa`` as the method.
"""

from methods.dfa import DenseFeedback

__all__ = ["DenseFeedback"]
