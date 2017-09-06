/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/dropout_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class DropoutOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto dims = ctx.Input<Tensor>("X")->dims();
    ctx.Output<Tensor>("Out")->Resize(dims);
    ctx.Output<Tensor>("Mask")->Resize(dims);
  }
};

class DropoutOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  DropoutOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<float>("dropout_prob", "Probability for dropping out units.")
        .SetDefault(.5f)
        .LargerThan(0.0f)
        .LessThan(1.0f);
    AddAttr<int>("seed", "Dropout random seed.").SetDefault(0);
    AddInput("X", "The input of dropout op.");
    AddOutput("Out", "The output of dropout op.");
    AddOutput("Mask", "The random sampled dropout mask.").AsIntermediate();

    AddComment(R"DOC(
Dropout Operator.

"Dropout" refers to randomly dropping out units in a nerual network. It is a
regularization technique for reducing overfitting by preventing neuron
co-adaption during training. The dropout operator randomly set (according to
the given dropout probability) the outputs of some units to zero, while others
being set to their inputs.
)DOC");
  }
};

class DropoutOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto mask_dims = ctx.Input<Tensor>("Mask")->dims();
    auto out_dims = ctx.Input<Tensor>(framework::GradVarName("Out"))->dims();
    PADDLE_ENFORCE_EQ(x_dims, out_dims,
                      "Dimensions of Input(X) and Out@Grad must be the same.");
    PADDLE_ENFORCE_EQ(x_dims, mask_dims,
                      "Dimensions of Input(X) and Mask must be the same.");
    // resize
    auto *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    x_grad->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(dropout, ops::DropoutOp, ops::DropoutOpMaker, dropout_grad,
            ops::DropoutOpGrad);
REGISTER_OP_CPU_KERNEL(
    dropout, ops::CPUDropoutKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    dropout_grad, ops::DropoutGradKernel<paddle::platform::CPUPlace, float>);
