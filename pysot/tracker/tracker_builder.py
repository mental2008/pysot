# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker
         }


def build_tracker(model, config):
    cfg.TRACK.PENALTY_K = float(config['penalty_k'])
    cfg.TRACK.WINDOW_INFLUENCE = float(config['window_influence'])
    cfg.TRACK.LR = float(config['lr'])
    cfg.TRACK.INSTANCE_SIZE = int(config['search_region'])
    return TRACKS[cfg.TRACK.TYPE](model)
