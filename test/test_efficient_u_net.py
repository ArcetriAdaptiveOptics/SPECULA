import unittest

# torch is an optional dependency (see pyproject.toml's "nn" extra): skip
# every test in this module rather than failing collection when it's absent.
try:
    import torch
    from iteras.lib.efficient_u_net import UNetRegressor, SpatialRegressionHead
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestUNetRegressorForward(unittest.TestCase):
    """Exercise the network actually used by Conv2dNetTrainer/Tester:
    every conv_block_type must build and produce the requested output shape."""

    def _build(self, conv_block_type=0, depth=2, base_channels=4, output_size=5, **kwargs):
        return UNetRegressor(
            input_channels=1,
            output_size=output_size,
            base_channels=base_channels,
            dropout_level=0.0,
            depth=depth,
            conv_block_type=conv_block_type,
            **kwargs,
        )

    def test_every_conv_block_type_forward_shape(self):
        for conv_block_type in (0, 1, 3):
            with self.subTest(conv_block_type=conv_block_type):
                out = self._build(conv_block_type=conv_block_type)(torch.randn(2, 1, 32, 32))
                self.assertEqual(out.shape, (2, 5))

    def test_forward_independent_of_input_spatial_size(self):
        # AdaptiveFeatureExtractor pools to a fixed spatial size, so the
        # regression head's input dimension - and thus the output shape -
        # does not depend on the actual input H, W.
        model = self._build()
        self.assertEqual(model(torch.randn(1, 1, 32, 32)).shape, (1, 5))
        self.assertEqual(model(torch.randn(1, 1, 64, 64)).shape, (1, 5))

    def test_forward_handles_odd_input_size_via_interpolation(self):
        # With an input size not evenly divisible by 2**depth, the decoder's
        # upsampled feature map and the corresponding encoder skip connection
        # end up with mismatched spatial sizes; the model must reconcile
        # them via interpolation instead of crashing.
        model = self._build(depth=2)
        self.assertEqual(model(torch.randn(1, 1, 33, 33)).shape, (1, 5))

    def test_check_input_size(self):
        UNetRegressor.check_input_size((8, 9), depth=3)
        with self.assertRaisesRegex(ValueError, 'too small for depth=4.*depth <= 3'):
            UNetRegressor.check_input_size((8, 9), depth=4)

    def test_small_bottleneck_attention(self):
        # A bottleneck with fewer than 8 channels keeps one query/key channel
        model = UNetRegressor(input_channels=1, output_size=3, base_channels=2, depth=0)
        self.assertEqual(model(torch.randn(2, 1, 8, 8)).shape, (2, 3))

    def test_unsupported_conv_block_type_raises(self):
        with self.assertRaisesRegex(ValueError, 'conv_block_type'):
            self._build(conv_block_type=99)

    def test_output_requires_grad_for_backward(self):
        model = self._build()
        model(torch.randn(2, 1, 32, 32)).sum().backward()
        # gradients must flow back to at least one trainable parameter
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestSpatialHead(unittest.TestCase):

    def _net(self, depth=3, **kwargs):
        return UNetRegressor(input_channels=8, output_size=20, base_channels=4,
                             depth=depth, head_type='spatial', **kwargs).eval()

    def test_output_shape_for_several_input_sizes(self):
        net = self._net()
        for size in [(60, 60), (32, 32), (15, 17), (8, 8)]:
            with torch.no_grad():
                out = net(torch.randn(2, 8, *size))
            self.assertEqual(tuple(out.shape), (2, 20), size)

    def test_last_decoder_map_keeps_the_input_resolution(self):
        # 60 -> 30 -> 15 -> 7: the odd level must be resized up to 15 on the
        # way back, not the skip connection down to 14.
        with torch.no_grad():
            _, decoder_outputs = self._net()._trunk(torch.randn(1, 8, 60, 60))
        self.assertEqual([tuple(d.shape[2:]) for d in decoder_outputs], [(15, 15), (30, 30), (60, 60)])

    def test_uses_spatial_head_and_no_pooled_extractors(self):
        net = self._net()
        self.assertIsInstance(net.regressor, SpatialRegressionHead)
        self.assertFalse(hasattr(net, 'feature_extractors'))

    def test_pooled_is_the_default(self):
        net = UNetRegressor(input_channels=1, output_size=5, base_channels=4, depth=2)
        self.assertEqual(net.head_type, 'pooled')
        self.assertNotIsInstance(net.regressor, SpatialRegressionHead)

    def test_head_grid_sets_the_head_resolution(self):
        net = self._net(head_grid=60)
        self.assertEqual(net.regressor.pool.output_size, 60)
        self.assertEqual(net.regressor.direct.in_features, 4 * 60 * 60)
        with torch.no_grad():
            self.assertEqual(tuple(net(torch.randn(2, 8, 120, 120)).shape), (2, 20))

    def test_invalid_head_type(self):
        with self.assertRaises(ValueError):
            UNetRegressor(head_type='nope')

    def test_can_represent_a_linear_map_of_the_pixels(self):
        # The direct path alone must be able to fit a linear target that
        # depends on where things are on the map (the pooled head, reducing
        # everything to <= 4x4 grids, can't tell nearby pixels apart).
        torch.manual_seed(0)
        net = UNetRegressor(input_channels=1, output_size=4, base_channels=4,
                            depth=2, dropout_level=0.0, head_type='spatial')
        x = torch.randn(512, 1, 16, 16)
        y = x.flatten(1) @ torch.randn(4, 256).T / 16
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(x), y)
            loss.backward()
            opt.step()
        self.assertLess(float(loss) / float(y.var()), 0.1)


if __name__ == '__main__':
    unittest.main()
