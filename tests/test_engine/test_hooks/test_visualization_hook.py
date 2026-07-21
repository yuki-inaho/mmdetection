# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import time
from unittest import TestCase
from unittest.mock import Mock

import torch
from mmengine.structures import InstanceData

from mmdet.engine.hooks import DetVisualizationHook, TrackVisualizationHook
from mmdet.engine.hooks.visualization_hook import \
    _restore_gt_instances_to_original_space
from mmdet.structures import DetDataSample, TrackDataSample
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.visualization import DetLocalVisualizer, TrackLocalVisualizer


def _rand_bboxes(num_boxes, h, w):
    cx, cy, bw, bh = torch.rand(num_boxes, 4).T

    tl_x = ((cx * w) - (w * bw / 2)).clamp(0, w)
    tl_y = ((cy * h) - (h * bh / 2)).clamp(0, h)
    br_x = ((cx * w) + (w * bw / 2)).clamp(0, w)
    br_y = ((cy * h) + (h * bh / 2)).clamp(0, h)

    bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=0).T
    return bboxes


class TestVisualizationHook(TestCase):

    def test_restore_gt_instances_to_original_space(self):
        """GT must match rescale=True predictions on the original image."""
        data_sample = DetDataSample()
        data_sample.set_metainfo(
            dict(ori_shape=(600, 800), img_shape=(512, 736),
                 scale_factor=(736 / 800, 512 / 600)))
        gt_instances = InstanceData()
        gt_instances.bboxes = HorizontalBoxes(
            [[92.0, 85.333333, 184.0, 170.666667]])
        gt_instances.labels = torch.tensor([0])
        data_sample.gt_instances = gt_instances

        display_sample = _restore_gt_instances_to_original_space(data_sample)

        self.assertIsNot(display_sample, data_sample)
        torch.testing.assert_close(
            display_sample.gt_instances.bboxes.tensor,
            torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
            rtol=0,
            atol=1e-4)
        # The metric receives the original runner output and must not observe
        # visualizer-only mutations.
        torch.testing.assert_close(
            data_sample.gt_instances.bboxes.tensor,
            torch.tensor([[92.0, 85.333333, 184.0, 170.666667]]),
            rtol=0,
            atol=1e-4)

    def setUp(self) -> None:
        DetLocalVisualizer.get_instance('current_visualizer')

        pred_instances = InstanceData()
        pred_instances.bboxes = _rand_bboxes(5, 10, 12)
        pred_instances.labels = torch.randint(0, 2, (5, ))
        pred_instances.scores = torch.rand((5, ))
        pred_det_data_sample = DetDataSample()
        pred_det_data_sample.set_metainfo({
            'img_path':
            osp.join(osp.dirname(__file__), '../../data/color.jpg')
        })
        pred_det_data_sample.pred_instances = pred_instances
        self.outputs = [pred_det_data_sample] * 2

    def test_after_val_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = DetVisualizationHook()
        hook.after_val_iter(runner, 1, {}, self.outputs)

    def test_after_test_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = DetVisualizationHook(draw=True)
        hook.after_test_iter(runner, 1, {}, self.outputs)
        self.assertEqual(hook._test_index, 2)

        # test
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        test_out_dir = timestamp + '1'
        runner.work_dir = timestamp
        runner.timestamp = '1'
        hook = DetVisualizationHook(draw=False, test_out_dir=test_out_dir)
        hook.after_test_iter(runner, 1, {}, self.outputs)
        self.assertTrue(not osp.exists(f'{timestamp}/1/{test_out_dir}'))

        hook = DetVisualizationHook(draw=True, test_out_dir=test_out_dir)
        hook.after_test_iter(runner, 1, {}, self.outputs)
        self.assertTrue(osp.exists(f'{timestamp}/1/{test_out_dir}'))
        shutil.rmtree(f'{timestamp}')


class TestTrackVisualizationHook(TestCase):

    def setUp(self) -> None:
        TrackLocalVisualizer.get_instance('visualizer')
        # pseudo data_batch
        self.data_batch = dict(data_samples=None, inputs=None)

        pred_instances_data = dict(
            bboxes=torch.tensor([[100, 100, 200, 200], [150, 150, 400, 200]]),
            instances_id=torch.tensor([1, 2]),
            labels=torch.tensor([0, 1]),
            scores=torch.tensor([0.955, 0.876]))
        pred_instances = InstanceData(**pred_instances_data)
        img_data_sample = DetDataSample()
        img_data_sample.pred_track_instances = pred_instances
        img_data_sample.gt_instances = pred_instances
        img_data_sample.set_metainfo(
            dict(
                img_path=osp.join(
                    osp.dirname(__file__), '../../data/color.jpg'),
                scale_factor=(1.0, 1.0)))
        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = [img_data_sample]
        track_data_sample.set_metainfo(dict(ori_length=1))
        self.outputs = [track_data_sample]

    def test_after_val_iter_image(self):
        runner = Mock()
        runner.iter = 1
        hook = TrackVisualizationHook(frame_interval=10, draw=True)
        hook.after_val_iter(runner, 9, self.data_batch, self.outputs)

    def test_after_test_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = TrackVisualizationHook(frame_interval=10, draw=True)
        hook.after_val_iter(runner, 9, self.data_batch, self.outputs)

        # test test_out_dir
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        test_out_dir = timestamp + '1'
        runner.work_dir = timestamp
        runner.timestamp = '1'
        hook = TrackVisualizationHook(
            frame_interval=10, draw=True, test_out_dir=test_out_dir)
        hook.after_test_iter(runner, 9, self.data_batch, self.outputs)
        self.assertTrue(osp.exists(f'{timestamp}/1/{test_out_dir}'))
        shutil.rmtree(f'{timestamp}')
