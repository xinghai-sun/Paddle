import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestCosSimOp2d(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "cos_sim"
        self.inputs = {
            'X': np.random.random((32, 64)).astype("float32"),
            'Y': np.random.random((32, 64)).astype("float32")
        }
        expect_x_norm = np.linalg.norm(self.inputs['X'], axis=1)
        expect_y_norm = np.linalg.norm(self.inputs['Y'], axis=1)
        expect_out = (self.inputs['X'] * self.inputs['Y']).sum(axis=1) / \
            expect_x_norm / expect_y_norm
        self.outputs = {
            'XNorm': np.expand_dims(expect_x_norm, 1),
            'YNorm': np.expand_dims(expect_y_norm, 1),
            'Out': np.expand_dims(expect_out, 1)
        }


class TestCosSimOp2dBcast(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "cos_sim"
        self.inputs = {
            'X': np.random.random((32, 64)).astype("float32"),
            'Y': np.random.random((1, 64)).astype("float32")
        }
        expect_x_norm = np.linalg.norm(self.inputs['X'], axis=1)
        expect_y_norm = np.linalg.norm(self.inputs['Y'], axis=1)
        expect_out = (self.inputs['X'] * self.inputs['Y']).sum(axis=1) / \
            expect_x_norm / expect_y_norm
        self.outputs = {
            'XNorm': np.expand_dims(expect_x_norm, 1),
            'YNorm': np.expand_dims(expect_y_norm, 1),
            'Out': np.expand_dims(expect_out, 1)
        }


class TestCosSimOp3dBcast(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "cos_sim"
        self.inputs = {
            'X': np.random.random((32, 64, 10)).astype("float32"),
            'Y': np.random.random((1, 64, 10)).astype("float32")
        }
        expect_x_norm = np.linalg.norm(self.inputs['X'], axis=(1, 2))
        expect_y_norm = np.linalg.norm(self.inputs['Y'], axis=(1, 2))
        expect_out = (self.inputs['X'] * self.inputs['Y']).sum(axis=(1, 2)) / \
            expect_x_norm / expect_y_norm
        self.outputs = {
            'XNorm': np.expand_dims(expect_x_norm, 1),
            'YNorm': np.expand_dims(expect_y_norm, 1),
            'Out': np.expand_dims(expect_out, 1)
        }


class TestCosSimGradOp(GradientChecker):
    def test_cos_sim_2d(self):
        op = create_op("cos_sim")
        inputs = {
            'X': np.random.random((6, 5)).astype("float32"),
            'Y': np.random.random((6, 5)).astype("float32")
        }
        self.compare_grad(op, inputs)
        self.check_grad(
            op, inputs, set(["X", "Y"]), "Out", max_relative_error=0.05)

    def test_cos_sim_2d_bcast(self):
        op = create_op("cos_sim")
        inputs = {
            'X': np.random.random((6, 5)).astype("float32"),
            'Y': np.random.random((1, 5)).astype("float32")
        }
        self.compare_grad(op, inputs)
        self.check_grad(
            op, inputs, set(["X", "Y"]), "Out", max_relative_error=0.05)

    def test_cos_sim_3d(self):
        op = create_op("cos_sim")
        inputs = {
            'X': np.random.random((6, 5, 2)).astype("float32"),
            'Y': np.random.random((6, 5, 2)).astype("float32")
        }
        self.compare_grad(op, inputs)
        self.check_grad(
            op, inputs, set(["X", "Y"]), "Out", max_relative_error=0.05)

    def test_cos_sim_3d_bcast(self):
        op = create_op("cos_sim")
        inputs = {
            'X': np.random.random((6, 5, 2)).astype("float32"),
            'Y': np.random.random((1, 5, 2)).astype("float32")
        }
        self.compare_grad(op, inputs)
        self.check_grad(
            op, inputs, set(["X", "Y"]), "Out", max_relative_error=0.05)


if __name__ == '__main__':
    unittest.main()
