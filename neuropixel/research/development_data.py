"""Restricted development adapter; historical ResearchRoleTask is unchanged.

Only sample() and frozen_dataset() using train/validation are supported here.
Final evaluation should consume an externally frozen artifact through FinalStudy;
this adapter never creates final examples. Public source is not an access-control
boundary and can still be called independently by arbitrary user code.
"""
from neuropixel.research.data import ResearchRoleTask


class DevelopmentRoleTask(ResearchRoleTask):
    def sample(self, batch, split="train", generator=None, device="cpu",
               query_role=None, place_pool=None, meta=False):
        if split not in ("train", "validation"):
            raise ValueError("development sampling accepts only train or validation")
        return super().sample(batch, split, generator, device, query_role, place_pool, meta)

    def sample_loop(self, *args, **kwargs):
        raise ValueError("legacy sample_loop is unsupported for governed development; use sample")

    def sample_camera(self, *args, **kwargs):
        raise ValueError("sample_camera requires its own governed protocol and is unsupported here")
