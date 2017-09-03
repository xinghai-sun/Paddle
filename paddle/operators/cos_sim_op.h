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

#pragma once
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class CosSimKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // get Tensor
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* z = context.Output<Tensor>("Out");
    auto* x_norm = context.Output<Tensor>("XNorm");
    auto* y_norm = context.Output<Tensor>("YNorm");
    z->mutable_data<T>(context.GetPlace());
    x_norm->mutable_data<T>(context.GetPlace());
    y_norm->mutable_data<T>(context.GetPlace());

    // convert Tensor to Eigen Tensor
    int rows_x = x->dims()[0];
    int rows_y = y->dims()[0];
    int cols = framework::product(x->dims()) / rows_x;
    auto X = EigenMatrix<T>::From(*x, framework::make_ddim({rows_x, cols}));
    auto Y = EigenMatrix<T>::From(*y, framework::make_ddim({rows_y, cols}));
    auto Z = EigenMatrix<T>::From(*z);
    auto XNorm = EigenMatrix<T>::From(*x_norm);
    auto YNorm = EigenMatrix<T>::From(*y_norm);

    // compute
    auto place = context.GetEigenDevice<Place>();
    XNorm.device(place) = (X * X).sum(Eigen::array<int, 1>({1})).sqrt();
    YNorm.device(place) = (Y * Y).sum(Eigen::array<int, 1>({1})).sqrt();
    if (rows_x == rows_y) {
      auto XY = (X * Y).sum(Eigen::array<int, 1>({1}));
      Z.device(place) = XY / XNorm / YNorm;
    } else {
      Eigen::DSizes<int, 2> bcast(rows_x, 1);
      auto XY = (X * Y.broadcast(bcast)).sum(Eigen::array<int, 1>({1}));
      Z.device(place) = XY / XNorm / YNorm.broadcast(bcast);
    }
  }
};

template <typename Place, typename T>
class CosSimGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // get Tensor
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* z = context.Input<Tensor>("Out");
    auto* x_norm = context.Input<Tensor>("XNorm");
    auto* y_norm = context.Input<Tensor>("YNorm");
    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Output<Tensor>(framework::GradVarName("Y"));
    auto* grad_z = context.Input<Tensor>(framework::GradVarName("Out"));
    grad_x->mutable_data<T>(context.GetPlace());
    grad_y->mutable_data<T>(context.GetPlace());

    // convert Tensor to Eigen Tensor
    int rows_x = x->dims()[0];
    int rows_y = y->dims()[0];
    int cols = framework::product(x->dims()) / rows_x;
    auto X = EigenMatrix<T>::From(*x, framework::make_ddim({rows_x, cols}));
    auto Y = EigenMatrix<T>::From(*y, framework::make_ddim({rows_y, cols}));
    auto Z = EigenMatrix<T>::From(*z);
    auto X_norm = EigenMatrix<T>::From(*x_norm);
    auto Y_norm = EigenMatrix<T>::From(*y_norm);
    auto dX =
        EigenMatrix<T>::From(*grad_x, framework::make_ddim({rows_x, cols}));
    auto dY =
        EigenMatrix<T>::From(*grad_y, framework::make_ddim({rows_y, cols}));
    auto dZ = EigenMatrix<T>::From(*grad_z);

    // compute gradident
    Eigen::DSizes<int, 2> bcast(1, cols);
    auto Z_bcast = Z.broadcast(bcast);
    auto dZ_bcast = dZ.broadcast(bcast);
    auto X_snorm_bcast = X_norm.square().eval().broadcast(bcast);
    auto place = context.GetEigenDevice<Place>();
    if (rows_x == rows_y) {
      auto Y_snorm_bcast = Y_norm.square().eval().broadcast(bcast);
      auto norm_prod_bcast = (X_norm * Y_norm).eval().broadcast(bcast);
      dX.device(place) =
          dZ_bcast * (Y / norm_prod_bcast - Z_bcast * X / X_snorm_bcast);
      dY.device(place) =
          dZ_bcast * (X / norm_prod_bcast - Z_bcast * Y / Y_snorm_bcast);
    } else {
      Eigen::DSizes<int, 2> bcast_row(rows_x, 1);
      auto Y_snorm_bcast =
          Y_norm.square().eval().broadcast(bcast_row).eval().broadcast(bcast);
      auto norm_prod_bcast =
          (X_norm * Y_norm.broadcast(bcast_row)).eval().broadcast(bcast);
      dX.device(place) = dZ_bcast * (Y.broadcast(bcast_row) / norm_prod_bcast -
                                     Z_bcast * X / X_snorm_bcast);
      dY.device(place) =
          (dZ_bcast * (X / norm_prod_bcast -
                       Z_bcast * Y.broadcast(bcast_row) / Y_snorm_bcast))
              .sum(Eigen::array<int, 1>({0}));
    }
  }
};

}  // namespace operators
}  // namespace paddle
