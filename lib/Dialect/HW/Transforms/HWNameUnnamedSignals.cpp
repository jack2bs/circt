//===- NameUnnamedSignals.cpp - Add wires for unnamed signals ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the NameUnnamedSignals pass. This pass traverses the hw
// modules and adds a wire for every signal that doesn't have a name.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWNAMEUNNAMEDSIGNALS
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {

struct HWNameUnnamedSignalsPass
    : public circt::hw::impl::HWNameUnnamedSignalsBase<HWNameUnnamedSignalsPass> {

  void runOnOperation() override;
};
} // end anonymous namespace

/// The main logic for the NameUnnamedSignals pass.
void HWNameUnnamedSignalsPass::runOnOperation() {

  auto module = getOperation();

  int i = 0;
  OpBuilder builder(module.getBodyRegion());

  module.walk([&](Operation *op) {
    if (!isCombinational(op))
    {
      return;
    }

    for (OpResult result : op->getResults()) {

      if (auto maybeNameLoc = result.getLoc()->findInstanceOf<mlir::NameLoc>())
        continue;

      // Generate a unique name for the wire.
      std::string wireName = "anon_" + std::to_string(i++);

      // Insert a wire for the unnamed signal.
      builder.setInsertionPointAfter(op);
      auto wire = builder.create<WireOp>(op->getLoc(), result.getType(), result,
                                          builder.getStringAttr(wireName), hw::InnerSymAttr::get(builder.getStringAttr(wireName)));
      result.replaceAllUsesExcept(wire->getResult(0), wire);
    }
  });
}